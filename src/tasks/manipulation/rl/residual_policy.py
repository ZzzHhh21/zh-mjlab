from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from rsl_rl.env.vec_env import VecEnv
from tensordict import TensorDict


@dataclass(frozen=True)
class ResidualPolicyConfig:
  """Runtime action composition for staged frozen-policy residual training."""

  residual_scale: float = 1.0
  base_policy_steps: int = 60
  contact_policy_steps: int = 0
  contact_scale: float = 1.0
  arm_residual_scale: float = 1.0
  hand_residual_scale: float = 1.0
  other_residual_scale: float = 1.0
  left_arm_residual_scale: float | None = None
  right_arm_residual_scale: float | None = None
  left_hand_residual_scale: float | None = None
  right_hand_residual_scale: float | None = None
  contact_arm_decay_steps: int = 0
  move_residual_right_arm_hand_only: bool = True
  move_freeze_left_arm_hand_to_contact: bool = True


class FrozenBasePolicy:
  """Callable wrapper that makes the loaded base policy inference-only."""

  def __init__(self, policy: Callable[[TensorDict], torch.Tensor]) -> None:
    self.policy = policy
    if hasattr(self.policy, "eval"):
      self.policy.eval()
    if hasattr(self.policy, "parameters"):
      for param in self.policy.parameters():
        param.requires_grad_(False)

  def __call__(self, obs: TensorDict) -> torch.Tensor:
    with torch.inference_mode():
      return self.policy(obs).detach()

  def eval(self) -> "FrozenBasePolicy":
    return self

  def parameters(self) -> tuple:
    return ()


class ResidualPolicyVecEnvWrapper(VecEnv):
  """Use frozen base/contact actions first, then train move residual actions."""

  def __init__(
    self,
    env: VecEnv,
    base_policy: Callable[[TensorDict], torch.Tensor],
    cfg: ResidualPolicyConfig,
    contact_policy: Callable[[TensorDict], torch.Tensor] | None = None,
  ) -> None:
    self.env = env
    self.base_policy = base_policy
    self.contact_policy = contact_policy
    self.residual_cfg = cfg

    self.num_envs = env.num_envs
    self._full_num_actions = int(env.num_actions)
    # Keep temporary full action dim while building masks/scales.
    self.num_actions = self._full_num_actions
    self.max_episode_length = env.max_episode_length
    self.device = torch.device(env.device)

    self._base_policy_steps = int(cfg.base_policy_steps)
    self._contact_policy_steps = int(cfg.contact_policy_steps)
    self._has_contact_policy = (
      self.contact_policy is not None and self._contact_policy_steps > 0
    )
    self._move_policy_start_step = (
      self._base_policy_steps + self._contact_policy_steps
      if self._has_contact_policy
      else self._base_policy_steps
    )
    self._step_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self._cached_base_action = torch.zeros(
      self.num_envs, self._full_num_actions, device=self.device
    )
    self._has_cached_base_action = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._cached_contact_action = torch.zeros(
      self.num_envs, self._full_num_actions, device=self.device
    )
    self._has_cached_contact_action = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._left_arm_hand_action_mask = self._build_arm_hand_action_mask("left")
    self._right_arm_hand_action_mask = self._build_arm_hand_action_mask("right")
    self._move_residual_scale = self._build_move_residual_scale()
    self._contact_reference_scale = self._build_contact_reference_scale()
    self._policy_action_ids = self._build_policy_action_ids()
    self.num_actions = int(self._policy_action_ids.numel())
    self._last_obs: TensorDict | None = None
    print(
      "[INFO] Residual policy action dim configured: "
      f"policy_dim={self.num_actions}, env_dim={self._full_num_actions}"
    )

    self._freeze_policy(self.base_policy)
    if self.contact_policy is not None:
      self._freeze_policy(self.contact_policy)

  @property
  def cfg(self):
    return self.env.cfg

  @property
  def render_mode(self) -> str | None:
    return self.env.render_mode

  @property
  def observation_space(self):
    return self.env.observation_space

  @property
  def action_space(self):
    return self.env.action_space

  @property
  def unwrapped(self):
    return self.env.unwrapped

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self.env.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:
    self.env.episode_length_buf = value

  def seed(self, seed: int = -1) -> int:
    return self.env.seed(seed)

  def get_observations(self) -> TensorDict:
    self._last_obs = self.env.get_observations()
    return self._last_obs

  def reset(self) -> tuple[TensorDict, dict]:
    obs, extras = self.env.reset()
    self._reset_state()
    self._last_obs = obs
    return obs, extras

  def step(
    self,
    actions: torch.Tensor,
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    obs = self._last_obs
    if obs is None:
      obs = self.get_observations()

    base_phase, contact_phase, move_phase = self._phase_masks()
    base_action = self._compute_base_actions(obs, base_phase, contact_phase, move_phase)
    final_actions = self._compose_actions(
      residual_actions=actions,
      obs=obs,
      base_actions=base_action,
      contact_phase=contact_phase,
      move_phase=move_phase,
    )
    residual_train_mask = move_phase.clone()
    next_obs, rewards, dones, extras = self.env.step(final_actions)
    extras = dict(extras)
    extras["residual_train_mask"] = residual_train_mask
    self._last_obs = next_obs
    self._step_counter += 1

    reset_mask = dones.to(dtype=torch.bool)
    if reset_mask.any():
      self._reset_state(reset_mask)

    return next_obs, rewards, dones, extras

  def close(self) -> None:
    return self.env.close()

  def _phase_masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_phase = self._step_counter < self._base_policy_steps
    if self._has_contact_policy:
      contact_phase = (
        (self._step_counter >= self._base_policy_steps)
        & (self._step_counter < self._move_policy_start_step)
      )
    else:
      contact_phase = torch.zeros_like(base_phase)
    move_phase = self._step_counter >= self._move_policy_start_step
    return base_phase, contact_phase, move_phase

  def _compute_base_actions(
    self,
    obs: TensorDict,
    base_phase: torch.Tensor,
    contact_phase: torch.Tensor,
    move_phase: torch.Tensor,
  ) -> torch.Tensor:
    base_action = torch.zeros(self.num_envs, self._full_num_actions, device=self.device)
    need_live_base = base_phase | (
      (contact_phase | move_phase) & ~self._has_cached_base_action
    )

    if need_live_base.any():
      with torch.inference_mode():
        base_action[need_live_base] = self.base_policy(obs[need_live_base]).detach()

    newly_cached = (contact_phase | move_phase) & ~self._has_cached_base_action
    if newly_cached.any():
      self._cached_base_action[newly_cached] = base_action[newly_cached]
      self._has_cached_base_action[newly_cached] = True

    cached_mask = self._has_cached_base_action & ~need_live_base
    if cached_mask.any():
      base_action[cached_mask] = self._cached_base_action[cached_mask]
    return base_action

  def _compose_actions(
    self,
    residual_actions: torch.Tensor,
    obs: TensorDict,
    base_actions: torch.Tensor,
    contact_phase: torch.Tensor,
    move_phase: torch.Tensor,
  ) -> torch.Tensor:
    clip_actions = getattr(self.env, "clip_actions", None)
    if clip_actions is not None:
      residual_actions = torch.clamp(residual_actions, -clip_actions, clip_actions)

    final_actions = base_actions.clone()

    if contact_phase.any():
      contact_action = self._compute_contact_actions(obs, contact_phase)
      contact_reference = self._reference_base_actions(base_actions)
      contact_final = (
        contact_reference
        + float(self.residual_cfg.contact_scale) * contact_action
      )
      final_actions[contact_phase] = contact_final[contact_phase]
      self._cached_contact_action[contact_phase] = contact_final[contact_phase]
      self._has_cached_contact_action[contact_phase] = True

    if move_phase.any():
      self._ensure_contact_action_cached(move_phase, obs, base_actions)
      move_reference = torch.where(
        self._has_cached_contact_action.unsqueeze(-1),
        self._cached_contact_action,
        self._reference_base_actions(base_actions),
      )
      move_reference = self._apply_contact_reference_decay(
        move_reference=move_reference,
        move_phase=move_phase,
      )
      if self.residual_cfg.move_freeze_left_arm_hand_to_contact:
        move_reference = self._freeze_left_arm_hand_contact_reference(
          move_reference=move_reference,
          move_phase=move_phase,
        )
      residual_actions_full = self._expand_policy_actions(residual_actions)
      residual_term = (
        float(self.residual_cfg.residual_scale)
        * self._move_residual_scale.unsqueeze(0)
        * residual_actions_full
      )
      move_final = (
        move_reference
        + residual_term
      )
      final_actions[move_phase] = move_final[move_phase]

    return final_actions

  def _apply_contact_reference_decay(
    self,
    move_reference: torch.Tensor,
    move_phase: torch.Tensor,
  ) -> torch.Tensor:
    decay_steps = int(self.residual_cfg.contact_arm_decay_steps)
    if decay_steps <= 0 or not move_phase.any():
      return move_reference

    move_steps = (self._step_counter - self._move_policy_start_step).clamp_min(0)
    arm_decay = (1.0 - move_steps.to(dtype=torch.float32) / float(decay_steps)).clamp(
      min=0.0,
      max=1.0,
    )
    reference_scale = torch.ones_like(move_reference)
    reference_scale[move_phase] = (
      self._contact_reference_scale.unsqueeze(0)
      + (1.0 - self._contact_reference_scale.unsqueeze(0))
      * arm_decay[move_phase].unsqueeze(-1)
    )
    return move_reference * reference_scale

  def _reference_base_actions(self, base_actions: torch.Tensor) -> torch.Tensor:
    return torch.where(
      self._has_cached_base_action.unsqueeze(-1),
      self._cached_base_action,
      base_actions,
    )

  def _freeze_left_arm_hand_contact_reference(
    self,
    move_reference: torch.Tensor,
    move_phase: torch.Tensor,
  ) -> torch.Tensor:
    if torch.count_nonzero(self._left_arm_hand_action_mask) == 0:
      return move_reference
    cached_contact_reference = torch.where(
      self._has_cached_contact_action.unsqueeze(-1),
      self._cached_contact_action,
      move_reference,
    )
    overwrite_mask = (
      move_phase.unsqueeze(-1) & self._left_arm_hand_action_mask.unsqueeze(0)
    )
    return torch.where(overwrite_mask, cached_contact_reference, move_reference)

  def _expand_policy_actions(self, policy_actions: torch.Tensor) -> torch.Tensor:
    expected_dim = int(self._policy_action_ids.numel())
    if policy_actions.shape[-1] != expected_dim:
      raise ValueError(
        "Policy action dimension mismatch: "
        f"expected {expected_dim}, got {policy_actions.shape[-1]}"
      )
    full_actions = torch.zeros(
      policy_actions.shape[0],
      self._full_num_actions,
      device=policy_actions.device,
      dtype=policy_actions.dtype,
    )
    full_actions[:, self._policy_action_ids] = policy_actions
    return full_actions

  def _compute_contact_actions(
    self,
    obs: TensorDict,
    mask: torch.Tensor,
  ) -> torch.Tensor:
    contact_actions = torch.zeros(
      self.num_envs, self._full_num_actions, device=self.device
    )
    if self.contact_policy is None or not mask.any():
      return contact_actions
    with torch.inference_mode():
      contact_actions[mask] = self.contact_policy(obs[mask]).detach()
    clip_actions = getattr(self.env, "clip_actions", None)
    if clip_actions is not None:
      contact_actions = torch.clamp(contact_actions, -clip_actions, clip_actions)
    return contact_actions

  def _ensure_contact_action_cached(
    self,
    move_phase: torch.Tensor,
    obs: TensorDict,
    base_actions: torch.Tensor,
  ) -> None:
    missing = move_phase & ~self._has_cached_contact_action
    if not missing.any():
      return

    if self._has_contact_policy:
      contact_action = self._compute_contact_actions(obs, missing)
      contact_reference = self._reference_base_actions(base_actions)
      self._cached_contact_action[missing] = (
        contact_reference[missing]
        + float(self.residual_cfg.contact_scale) * contact_action[missing]
      )
    else:
      self._cached_contact_action[missing] = self._reference_base_actions(
        base_actions
      )[missing]
    self._has_cached_contact_action[missing] = True

  def _build_arm_hand_action_mask(self, side: str) -> torch.Tensor:
    mask = torch.zeros(self._full_num_actions, device=self.device, dtype=torch.bool)
    target_names = self._get_action_target_names()
    if target_names is None or len(target_names) != self._full_num_actions:
      return mask
    prefix = f"{side}_"
    for action_id, target_name in enumerate(target_names):
      if not target_name.startswith(prefix):
        continue
      if "_finger" in target_name or any(
        token in target_name for token in ("_shoulder", "_elbow", "_wrist")
      ):
        mask[action_id] = True
    return mask

  def _build_policy_action_ids(self) -> torch.Tensor:
    if (
      self.residual_cfg.move_residual_right_arm_hand_only
      and torch.count_nonzero(self._right_arm_hand_action_mask) > 0
    ):
      return torch.nonzero(self._right_arm_hand_action_mask, as_tuple=False).squeeze(-1)
    return torch.arange(self._full_num_actions, device=self.device, dtype=torch.long)

  def _build_move_residual_scale(self) -> torch.Tensor:
    scale = torch.full(
      (self._full_num_actions,),
      float(self.residual_cfg.other_residual_scale),
      device=self.device,
      dtype=torch.float32,
    )
    target_names = self._get_action_target_names()
    if target_names is None or len(target_names) != self._full_num_actions:
      return scale

    for action_id, target_name in enumerate(target_names):
      if "_finger" in target_name:
        scale[action_id] = self._side_scale(
          target_name,
          left_scale=self.residual_cfg.left_hand_residual_scale,
          right_scale=self.residual_cfg.right_hand_residual_scale,
          fallback_scale=self.residual_cfg.hand_residual_scale,
        )
      elif any(
        token in target_name
        for token in ("_shoulder", "_elbow", "_wrist")
      ):
        scale[action_id] = self._side_scale(
          target_name,
          left_scale=self.residual_cfg.left_arm_residual_scale,
          right_scale=self.residual_cfg.right_arm_residual_scale,
          fallback_scale=self.residual_cfg.arm_residual_scale,
        )
    return scale

  def _build_contact_reference_scale(self) -> torch.Tensor:
    scale = torch.ones(self._full_num_actions, device=self.device, dtype=torch.float32)
    target_names = self._get_action_target_names()
    if target_names is None or len(target_names) != self._full_num_actions:
      return scale

    for action_id, target_name in enumerate(target_names):
      if any(
        token in target_name
        for token in ("_shoulder", "_elbow", "_wrist", "_finger")
      ):
        scale[action_id] = 0.0
    return scale

  @staticmethod
  def _side_scale(
    target_name: str,
    *,
    left_scale: float | None,
    right_scale: float | None,
    fallback_scale: float,
  ) -> float:
    if target_name.startswith("left_") and left_scale is not None:
      return float(left_scale)
    if target_name.startswith("right_") and right_scale is not None:
      return float(right_scale)
    return float(fallback_scale)

  def _get_action_target_names(self) -> tuple[str, ...] | None:
    unwrapped = getattr(self.env, "unwrapped", self.env)
    action_manager = getattr(unwrapped, "action_manager", None)
    if action_manager is None:
      return None
    try:
      action_term = action_manager.get_term("joint_pos")
    except Exception:
      return None
    target_names = getattr(action_term, "target_names", None)
    if target_names is None:
      return None
    names = tuple(str(name) for name in target_names)
    if len(names) != self._full_num_actions:
      return None
    return names

  @staticmethod
  def _freeze_policy(policy: Callable[[TensorDict], torch.Tensor]) -> None:
    if hasattr(policy, "eval"):
      policy.eval()
    if hasattr(policy, "parameters"):
      for param in policy.parameters():
        param.requires_grad_(False)

  def _reset_state(self, env_mask: torch.Tensor | None = None) -> None:
    if env_mask is None:
      self._step_counter.zero_()
      self._cached_base_action.zero_()
      self._has_cached_base_action.zero_()
      self._cached_contact_action.zero_()
      self._has_cached_contact_action.zero_()
      return

    self._step_counter[env_mask] = 0
    self._cached_base_action[env_mask] = 0.0
    self._has_cached_base_action[env_mask] = False
    self._cached_contact_action[env_mask] = 0.0
    self._has_cached_contact_action[env_mask] = False
