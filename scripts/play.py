"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mjlab
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from src.tasks.manipulation.rl.residual_policy import (
  FrozenBasePolicy,
  ResidualPolicyConfig,
  ResidualPolicyVecEnvWrapper,
)


def _scale_or_default(value: float | None, fallback: float) -> float:
  return float(fallback if value is None else value)


def _sync_residual_train_start_with_env_cfg(
  env_cfg,
  residual_train_start_step: int,
) -> None:
  terminations = getattr(env_cfg, "terminations", None)
  if not terminations:
    return
  term_cfg = terminations.get("key_fingertip_target_distance_failure")
  if term_cfg is None:
    return
  params = dict(getattr(term_cfg, "params", {}) or {})
  params["active_after_steps"] = max(
    int(params.get("active_after_steps", 0)),
    int(residual_train_start_step),
  )
  term_cfg.params = params


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
  return tensor.detach().cpu().clone()


def _root_state(entity) -> torch.Tensor:
  return torch.cat(
    (
      entity.data.root_link_pos_w,
      entity.data.root_link_quat_w,
      entity.data.root_link_lin_vel_w,
      entity.data.root_link_ang_vel_w,
    ),
    dim=-1,
  )


def _action_target_names(env) -> tuple[str, ...] | None:
  action_manager = getattr(env.unwrapped, "action_manager", None)
  if action_manager is None:
    return None
  try:
    action_term = action_manager.get_term("joint_pos")
  except Exception:
    return None
  target_names = getattr(action_term, "target_names", None)
  if target_names is None:
    return None
  return tuple(str(name) for name in target_names)


class PlayStateRecorder:
  """Record play-time states without changing env stepping semantics."""

  def __init__(
    self,
    env,
    *,
    output_file: str,
    include_observations: bool = False,
  ) -> None:
    self.env = env
    self.output_file = Path(output_file).expanduser()
    self.include_observations = bool(include_observations)
    self.num_envs = env.num_envs
    self.num_actions = getattr(
      env,
      "num_actions",
      env.unwrapped.action_manager.total_action_dim,
    )
    self.max_episode_length = getattr(env, "max_episode_length", None)
    self.device = torch.device(env.device)
    self._storage: dict[str, list[torch.Tensor]] = {}
    self._saved = False
    self._closed = False

    obs = self.env.get_observations()
    self._append_snapshot(
      action=None,
      reward=None,
      done=None,
      obs=obs if self.include_observations else None,
      is_reset=True,
    )

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

  def __getattr__(self, name: str):
    return getattr(self.env, name)

  def get_observations(self):
    return self.env.get_observations()

  def reset(self):
    result = self.env.reset()
    obs = result[0] if isinstance(result, tuple) else None
    self._append_snapshot(
      action=None,
      reward=None,
      done=torch.ones(self.num_envs, device=self.device, dtype=torch.bool),
      obs=obs if self.include_observations else None,
      is_reset=True,
    )
    return result

  def step(self, actions: torch.Tensor):
    obs, reward, done, extras = self.env.step(actions)
    self._append_snapshot(
      action=actions,
      reward=reward,
      done=done,
      obs=obs if self.include_observations else None,
      is_reset=False,
    )
    return obs, reward, done, extras

  def save(self) -> None:
    if self._saved:
      return
    self.output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
      "metadata": {
        "num_envs": int(self.num_envs),
        "num_actions": int(self.num_actions),
        "include_observations": self.include_observations,
        "action_target_names": _action_target_names(self.env),
      },
      "data": {
        key: torch.stack(values, dim=0)
        for key, values in self._storage.items()
      },
    }
    torch.save(payload, self.output_file)
    self._saved = True
    print(f"[INFO]: Saved play states to: {self.output_file}")

  def close(self) -> None:
    self.save()
    if not self._closed:
      self.env.close()
      self._closed = True

  def _append_snapshot(
    self,
    *,
    action: torch.Tensor | None,
    reward: torch.Tensor | None,
    done: torch.Tensor | None,
    obs,
    is_reset: bool,
  ) -> None:
    unwrapped = self.env.unwrapped
    robot = unwrapped.scene["robot"]
    left_obj = unwrapped.scene["left_obj"]
    right_obj = unwrapped.scene["right_obj"]

    if action is None:
      action = torch.zeros(self.num_envs, self.num_actions, device=self.device)
    if reward is None:
      reward = torch.zeros(self.num_envs, device=self.device)
    if done is None:
      done = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    self._storage.setdefault("robot_joint_pos", []).append(
      _cpu_clone(robot.data.joint_pos)
    )
    self._storage.setdefault("robot_joint_vel", []).append(
      _cpu_clone(robot.data.joint_vel)
    )
    self._storage.setdefault("robot_root_state", []).append(
      _cpu_clone(_root_state(robot))
    )
    self._storage.setdefault("left_obj_root_state", []).append(
      _cpu_clone(_root_state(left_obj))
    )
    self._storage.setdefault("right_obj_root_state", []).append(
      _cpu_clone(_root_state(right_obj))
    )
    self._storage.setdefault("episode_length", []).append(
      _cpu_clone(unwrapped.episode_length_buf)
    )
    self._storage.setdefault("env_origins", []).append(
      _cpu_clone(unwrapped.scene.env_origins)
    )
    self._storage.setdefault("action", []).append(_cpu_clone(action))
    self._storage.setdefault("reward", []).append(_cpu_clone(reward))
    self._storage.setdefault("done", []).append(
      _cpu_clone(done.to(dtype=torch.bool))
    )
    self._storage.setdefault("is_reset", []).append(
      torch.full((self.num_envs,), bool(is_reset), dtype=torch.bool)
    )

    if obs is None:
      return
    for key, value in obs.items():
      if torch.is_tensor(value):
        self._storage.setdefault(f"obs/{key}", []).append(_cpu_clone(value))


class StageAStateInitEnvWrapper:
  """Reset play envs from recorded robot joint states and object root states."""

  def __init__(
    self,
    env,
    *,
    state_file: str,
    start_frame: int | None = None,
    end_frame: int | None = None,
  ) -> None:
    self.env = env
    self.state_file = Path(state_file).expanduser()
    if not self.state_file.exists():
      raise FileNotFoundError(f"Stage-A init state file not found: {self.state_file}")

    payload = torch.load(self.state_file, map_location="cpu")
    data = payload.get("data", payload)
    data = self._slice_recorded_frames(data, start_frame, end_frame)
    if "robot_joint_pos" not in data:
      raise KeyError(f"{self.state_file} does not contain data['robot_joint_pos']")
    self._robot_joint_pos = self._flatten_state_tensor(data["robot_joint_pos"])
    self._robot_joint_vel = self._flatten_state_tensor(
      data.get("robot_joint_vel", torch.zeros_like(data["robot_joint_pos"]))
    )
    if self._robot_joint_pos.shape != self._robot_joint_vel.shape:
      raise ValueError(
        "robot_joint_pos and robot_joint_vel must have identical shapes after flattening."
      )

    robot = self.unwrapped.scene["robot"]
    if self._robot_joint_pos.shape[-1] != robot.data.joint_pos.shape[-1]:
      raise ValueError(
        "Recorded robot_joint_pos dimension does not match current robot joints: "
        f"{self._robot_joint_pos.shape[-1]} vs {robot.data.joint_pos.shape[-1]}"
      )

    self._device = torch.device(self.unwrapped.device)
    self._robot_joint_pos = self._robot_joint_pos.to(self._device)
    self._robot_joint_vel = self._robot_joint_vel.to(self._device)
    self._num_states = int(self._robot_joint_pos.shape[0])
    if self._num_states <= 0:
      raise ValueError(f"No states found in {self.state_file}")

    self._source_env_origins = self._load_env_origins(data, self._num_states).to(
      self.device
    )
    self._left_obj_root_state = self._load_optional_root_state(
      data, "left_obj_root_state"
    )
    self._right_obj_root_state = self._load_optional_root_state(
      data, "right_obj_root_state"
    )
    if (self._left_obj_root_state is None) != (self._right_obj_root_state is None):
      raise ValueError(
        "State file must contain both left_obj_root_state and right_obj_root_state, "
        "or neither."
      )
    self._has_object_root_states = self._left_obj_root_state is not None
    if self._left_obj_root_state is not None:
      self._left_obj_root_state = self._left_obj_root_state.to(self.device)
      self._right_obj_root_state = self._right_obj_root_state.to(self.device)

    print(
      "[INFO]: Stage-A init state bank enabled for play: "
      f"file={self.state_file}, states={self._num_states}, "
      f"joint_dim={self._robot_joint_pos.shape[-1]}, "
      f"object_root_states={self._has_object_root_states}, "
      f"frame_range=[{self._frame_start}, {self._frame_end})"
    )

  @property
  def unwrapped(self):
    return self.env.unwrapped

  @property
  def device(self) -> torch.device:
    return self._device

  def __getattr__(self, name: str):
    return getattr(self.env, name)

  def reset(self, **kwargs):
    obs, extras = self.env.reset(**kwargs)
    env_ids = kwargs.get("env_ids")
    if env_ids is None:
      env_ids = torch.arange(self.unwrapped.num_envs, device=self.device)
    obs = self._apply_random_stage_a_states(env_ids)
    return obs, extras

  def step(self, action: torch.Tensor):
    obs, reward, terminated, truncated, extras = self.env.step(action)
    reset_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
    if reset_env_ids.numel() > 0:
      obs = self._apply_random_stage_a_states(reset_env_ids)
    return obs, reward, terminated, truncated, extras

  def close(self) -> None:
    self.env.close()

  def _apply_random_stage_a_states(self, env_ids: torch.Tensor) -> dict:
    env_ids = env_ids.to(device=self.device, dtype=torch.long)
    sample_ids = torch.randint(
      low=0,
      high=self._num_states,
      size=(env_ids.numel(),),
      device=self.device,
    )
    joint_pos = self._robot_joint_pos[sample_ids]
    joint_vel = self._robot_joint_vel[sample_ids]

    robot = self.unwrapped.scene["robot"]
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)

    if self._has_object_root_states:
      env_origins = self.unwrapped.scene.env_origins[env_ids]
      left_obj = self.unwrapped.scene["left_obj"]
      right_obj = self.unwrapped.scene["right_obj"]
      left_obj.write_root_state_to_sim(
        self._root_state_for_target_env(
          self._left_obj_root_state,
          sample_ids,
          env_origins,
        ),
        env_ids=env_ids,
      )
      right_obj.write_root_state_to_sim(
        self._root_state_for_target_env(
          self._right_obj_root_state,
          sample_ids,
          env_origins,
        ),
        env_ids=env_ids,
      )

    self.unwrapped.scene.write_data_to_sim()
    self.unwrapped.sim.forward()
    self.unwrapped.observation_manager.reset(env_ids)
    self.unwrapped.sim.sense()
    self.unwrapped.obs_buf = self.unwrapped.observation_manager.compute(
      update_history=True
    )
    return self.unwrapped.obs_buf

  def _root_state_for_target_env(
    self,
    state_bank: torch.Tensor,
    sample_ids: torch.Tensor,
    target_env_origins: torch.Tensor,
  ) -> torch.Tensor:
    root_state = state_bank[sample_ids].clone()
    source_env_origins = self._source_env_origins[sample_ids]
    root_state[:, :3] = root_state[:, :3] - source_env_origins + target_env_origins
    return root_state

  @staticmethod
  def _flatten_state_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tensor):
      raise TypeError("Recorded state values must be torch.Tensor")
    if tensor.ndim < 2:
      raise ValueError(f"Expected state tensor with at least 2 dims, got {tensor.shape}")
    return tensor.reshape(-1, tensor.shape[-1]).to(dtype=torch.float32)

  def _load_optional_root_state(
    self,
    data: dict,
    key: str,
  ) -> torch.Tensor | None:
    if key not in data:
      return None
    root_state = self._flatten_state_tensor(data[key])
    if root_state.shape[0] != self._num_states or root_state.shape[-1] != 13:
      raise ValueError(
        f"{key} must flatten to shape ({self._num_states}, 13), "
        f"got {tuple(root_state.shape)}"
      )
    return root_state

  @staticmethod
  def _load_env_origins(data: dict, num_states: int) -> torch.Tensor:
    if "env_origins" not in data:
      return torch.zeros(num_states, 3, dtype=torch.float32)
    origins = StageAStateInitEnvWrapper._flatten_state_tensor(data["env_origins"])
    if origins.shape[0] != num_states or origins.shape[-1] != 3:
      raise ValueError(
        f"env_origins must flatten to shape ({num_states}, 3), "
        f"got {tuple(origins.shape)}"
      )
    return origins

  def _slice_recorded_frames(
    self,
    data: dict,
    start_frame: int | None,
    end_frame: int | None,
  ) -> dict:
    robot_joint_pos = data.get("robot_joint_pos")
    if not torch.is_tensor(robot_joint_pos) or robot_joint_pos.ndim < 3:
      self._frame_start = None
      self._frame_end = None
      return data

    num_frames = int(robot_joint_pos.shape[0])
    start = 0 if start_frame is None else int(start_frame)
    end = num_frames if end_frame is None else int(end_frame)
    if start < 0:
      start += num_frames
    if end < 0:
      end += num_frames
    if not (0 <= start < end <= num_frames):
      raise ValueError(
        f"Invalid recorded frame range [{start}, {end}) for {num_frames} frames."
      )

    self._frame_start = start
    self._frame_end = end
    sliced = {}
    for key, value in data.items():
      if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == num_frames:
        sliced[key] = value[start:end]
      else:
        sliced[key] = value
    return sliced


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  residual_base_checkpoint_file: str | None = None
  residual_contact_checkpoint_file: str | None = None
  residual_scale: float = 1.0
  residual_base_steps: int = 60
  residual_contact_steps: int = 0
  residual_contact_scale: float = 1.0
  residual_arm_scale: float = 1.0
  residual_hand_scale: float = 1.0
  residual_other_scale: float = 1.0
  residual_left_arm_scale: float | None = None
  residual_right_arm_scale: float | None = None
  residual_left_hand_scale: float | None = None
  residual_right_hand_scale: float | None = None
  residual_contact_arm_decay_steps: int = 0
  stage_a_init_state_file: str | None = None
  """Use recorded robot/object states as play reset states."""
  stage_a_init_start_frame: int | None = None
  """Optional first recorded frame to sample from in stage_a_init_state_file."""
  stage_a_init_end_frame: int | None = None
  """Optional exclusive recorded frame end to sample from in stage_a_init_state_file."""
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  stage_a_state_output_file: str | None = None
  """Save play-time robot/object states to this .pt file."""
  stage_a_state_max_steps: int | None = None
  """Stop play after this many sim steps; useful for recording only stage A."""
  stage_a_state_include_observations: bool = False
  """Also save observations in the state file. Off by default to keep files small."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def _load_frozen_policy(
  task_id: str,
  env,
  agent_cfg,
  device: str,
  checkpoint_file: str,
  policy_label: str,
):
  checkpoint_path = Path(checkpoint_file).expanduser()
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"{policy_label} checkpoint not found: {checkpoint_path}")

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  base_runner = runner_cls(env, asdict(agent_cfg), device=device)
  base_runner.load(
    str(checkpoint_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  print(f"[INFO]: Frozen {policy_label} policy loaded from: {checkpoint_path}")
  return FrozenBasePolicy(base_runner.get_inference_policy(device=device))


def _residual_train_start_step(cfg: PlayConfig) -> int:
  if cfg.residual_contact_checkpoint_file is None:
    return int(cfg.residual_base_steps)
  return int(cfg.residual_base_steps) + int(cfg.residual_contact_steps)


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE
  residual_mode = cfg.residual_base_checkpoint_file is not None
  if cfg.residual_contact_checkpoint_file is not None and not residual_mode:
    raise ValueError(
      "--residual-contact-checkpoint-file requires --residual-base-checkpoint-file."
    )
  if (
    cfg.residual_contact_checkpoint_file is not None
    and cfg.residual_contact_steps <= 0
  ):
    raise ValueError(
      "--residual-contact-steps must be > 0 when "
      "--residual-contact-checkpoint-file is set."
    )
  if cfg.residual_contact_checkpoint_file is None and cfg.residual_contact_steps > 0:
    raise ValueError(
      "--residual-contact-steps requires --residual-contact-checkpoint-file."
    )
  if cfg.stage_a_init_state_file is not None:
    if not residual_mode:
      raise ValueError("--stage-a-init-state-file requires residual play mode.")
    if cfg.residual_contact_checkpoint_file is None:
      raise ValueError(
        "--stage-a-init-state-file is intended for direct B->C play and requires "
        "--residual-contact-checkpoint-file."
      )
    if cfg.residual_base_steps != 0:
      raise ValueError(
        "--stage-a-init-state-file skips stage A, so set --residual-base-steps=0."
      )
  elif (
    cfg.stage_a_init_start_frame is not None
    or cfg.stage_a_init_end_frame is not None
  ):
    raise ValueError(
      "--stage-a-init-start-frame/--stage-a-init-end-frame require "
      "--stage-a-init-state-file."
    )

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")
  elif residual_mode:
    _sync_residual_train_start_with_env_cfg(
      env_cfg,
      _residual_train_start_step(cfg),
    )

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  if cfg.stage_a_init_state_file is not None:
    env = StageAStateInitEnvWrapper(
      env,
      state_file=cfg.stage_a_init_state_file,
      start_frame=cfg.stage_a_init_start_frame,
      end_frame=cfg.stage_a_init_end_frame,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if residual_mode:
    base_policy = _load_frozen_policy(
      task_id=task_id,
      env=env,
      agent_cfg=agent_cfg,
      device=device,
      checkpoint_file=cfg.residual_base_checkpoint_file,
      policy_label="base",
    )
    contact_policy = None
    if cfg.residual_contact_checkpoint_file is not None:
      contact_policy = _load_frozen_policy(
        task_id=task_id,
        env=env,
        agent_cfg=agent_cfg,
        device=device,
        checkpoint_file=cfg.residual_contact_checkpoint_file,
        policy_label="contact",
      )
    env = ResidualPolicyVecEnvWrapper(
      env,
      base_policy=base_policy,
      contact_policy=contact_policy,
      cfg=ResidualPolicyConfig(
        residual_scale=cfg.residual_scale,
        base_policy_steps=cfg.residual_base_steps,
        contact_policy_steps=cfg.residual_contact_steps,
        contact_scale=cfg.residual_contact_scale,
        arm_residual_scale=cfg.residual_arm_scale,
        hand_residual_scale=cfg.residual_hand_scale,
        other_residual_scale=cfg.residual_other_scale,
        left_arm_residual_scale=cfg.residual_left_arm_scale,
        right_arm_residual_scale=cfg.residual_right_arm_scale,
        left_hand_residual_scale=cfg.residual_left_hand_scale,
        right_hand_residual_scale=cfg.residual_right_hand_scale,
        contact_arm_decay_steps=cfg.residual_contact_arm_decay_steps,
      ),
    )
    print(
      "[INFO]: Residual policy mode enabled: "
      f"base_steps={cfg.residual_base_steps}, "
      f"contact_steps={cfg.residual_contact_steps}, "
      f"contact_scale={cfg.residual_contact_scale}, "
      f"contact_arm_decay_steps={cfg.residual_contact_arm_decay_steps}, "
      f"move_scale={cfg.residual_scale}, "
      f"arm_scale={cfg.residual_arm_scale}, "
      f"hand_scale={cfg.residual_hand_scale}, "
      f"other_scale={cfg.residual_other_scale}, "
      f"left_arm_scale={_scale_or_default(cfg.residual_left_arm_scale, cfg.residual_arm_scale)}, "
      f"right_arm_scale={_scale_or_default(cfg.residual_right_arm_scale, cfg.residual_arm_scale)}, "
      f"left_hand_scale={_scale_or_default(cfg.residual_left_hand_scale, cfg.residual_hand_scale)}, "
      f"right_hand_scale={_scale_or_default(cfg.residual_right_hand_scale, cfg.residual_hand_scale)}, "
      f"train_start_step={_residual_train_start_step(cfg)}"
    )
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  if cfg.stage_a_state_output_file is not None:
    env = PlayStateRecorder(
      env,
      output_file=cfg.stage_a_state_output_file,
      include_observations=cfg.stage_a_state_include_observations,
    )
    print(
      "[INFO]: Play state recording enabled: "
      f"output={cfg.stage_a_state_output_file}, "
      f"max_steps={cfg.stage_a_state_max_steps}"
    )

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  run_steps = cfg.stage_a_state_max_steps
  try:
    if resolved_viewer == "native":
      NativeMujocoViewer(env, policy).run(num_steps=run_steps)
    elif resolved_viewer == "viser":
      ViserPlayViewer(env, policy).run(num_steps=run_steps)
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  finally:
    env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
