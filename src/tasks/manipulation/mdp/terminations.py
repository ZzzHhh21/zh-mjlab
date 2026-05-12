from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_mul,
)
from mjlab.utils.nan_guard import NanGuard

from .rewards import (
  _active_waypoint_position,
  _pairwise_site_distances,
  _right_hand_grasp_gate_mask,
  _right_object_stage_distances,
  _right_object_stage_positions,
  _right_object_stage_waypoints,
  _repeat_stage_values_by_interp,
  _update_gate_state,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.termination_manager import TerminationTermCfg


def _has_nonfinite(values: torch.Tensor) -> torch.Tensor:
  if values.numel() == 0:
    return torch.zeros(values.shape[0], dtype=torch.bool, device=values.device)
  return ~torch.isfinite(values).flatten(start_dim=1).all(dim=1)


def _has_abs_over(values: torch.Tensor, limit: float | None) -> torch.Tensor:
  if limit is None or values.numel() == 0:
    return torch.zeros(values.shape[0], dtype=torch.bool, device=values.device)
  return values.abs().flatten(start_dim=1).gt(limit).any(dim=1)


def invalid_physics_state(
  env: ManagerBasedRlEnv,
  asset_names: tuple[str, ...] = ("robot",),
  max_joint_pos_abs: float | None = 20.0,
  max_joint_vel_abs: float | None = 500.0,
  max_root_distance: float | None = 5.0,
  max_root_vel_abs: float | None = 50.0,
  max_body_distance: float | None = 5.0,
) -> torch.Tensor:
  """Terminate envs with NaN/Inf or clearly exploded physics state."""
  bad = NanGuard.detect_nans(env.sim.data).clone()
  env_origins = env.scene.env_origins

  for asset_name in asset_names:
    asset: Entity = env.scene[asset_name]
    data = asset.data

    if data.is_articulated:
      joint_pos = data.joint_pos
      joint_vel = data.joint_vel
      bad |= _has_nonfinite(joint_pos)
      bad |= _has_nonfinite(joint_vel)
      bad |= _has_abs_over(joint_pos, max_joint_pos_abs)
      bad |= _has_abs_over(joint_vel, max_joint_vel_abs)

    free_q_adr = data.indexing.free_joint_q_adr
    free_v_adr = data.indexing.free_joint_v_adr
    if free_q_adr.numel() >= 7:
      root_pos = data.data.qpos[:, free_q_adr[:3]]
      root_quat = data.data.qpos[:, free_q_adr[3:7]]
      bad |= _has_nonfinite(root_pos)
      bad |= _has_nonfinite(root_quat)
      if max_root_distance is not None:
        root_pos_local = root_pos - env_origins
        bad |= torch.linalg.norm(root_pos_local, dim=-1).gt(max_root_distance)

      root_vel = data.data.qvel[:, free_v_adr]
      bad |= _has_nonfinite(root_vel)
      bad |= _has_abs_over(root_vel, max_root_vel_abs)

    if max_body_distance is not None:
      body_pos_local = data.body_link_pos_w - env_origins[:, None, :]
      bad |= _has_nonfinite(body_pos_local)
      bad |= torch.linalg.norm(body_pos_local, dim=-1).amax(dim=1).gt(
        max_body_distance
      )

  return bad


def object_rotation_over_limit(
  env: ManagerBasedRlEnv,
  asset_names: tuple[str, ...],
  max_angle: float,
  reference_euler_xyz_by_asset: dict[str, tuple[float, float, float]] | None = None,
) -> torch.Tensor:
  """Terminate when an object's root orientation deviates too far from reset pose."""
  bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  for asset_name in asset_names:
    asset: Entity = env.scene[asset_name]
    data = asset.data
    free_q_adr = data.indexing.free_joint_q_adr
    if free_q_adr.numel() < 7:
      continue

    current_quat = data.data.qpos[:, free_q_adr[3:7]]
    bad |= _has_nonfinite(current_quat)

    reference_quat = data.default_root_state[:, 3:7]
    if reference_euler_xyz_by_asset is not None:
      reference_euler = reference_euler_xyz_by_asset.get(asset_name)
      if reference_euler is not None:
        roll, pitch, yaw = reference_euler
        n = env.num_envs
        delta_quat = quat_from_euler_xyz(
          torch.full((n,), roll, device=env.device),
          torch.full((n,), pitch, device=env.device),
          torch.full((n,), yaw, device=env.device),
        )
        reference_quat = quat_mul(reference_quat, delta_quat)

    safe_current_quat = torch.nan_to_num(
      current_quat, nan=0.0, posinf=0.0, neginf=0.0
    )
    angle = quat_error_magnitude(safe_current_quat, reference_quat)
    bad |= torch.nan_to_num(angle, nan=float("inf")).gt(max_angle)

  return bad


def key_fingertip_target_distance_over_limit(
  env: ManagerBasedRlEnv,
  left_fingertip_cfg: SceneEntityCfg,
  left_target_cfg: SceneEntityCfg,
  right_fingertip_cfg: SceneEntityCfg,
  right_target_cfg: SceneEntityCfg,
  distance_threshold: float,
  active_after_steps: int = 0,
) -> torch.Tensor:
  """Terminate when any key fingertip leaves the target neighborhood."""
  if active_after_steps > 0:
    active = env.episode_length_buf >= int(active_after_steps)
  else:
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  left_dist = _pairwise_site_distances(env, left_fingertip_cfg, left_target_cfg)
  right_dist = _pairwise_site_distances(env, right_fingertip_cfg, right_target_cfg)
  over_limit = left_dist.gt(distance_threshold).any(dim=-1) | right_dist.gt(
    distance_threshold
  ).any(dim=-1)
  return active & over_limit


class right_object_stage_success:
  """Terminate once the right object reaches the final stage target."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    self.env = env
    self.activation_hold_steps = int(cfg.params["activation_hold_steps"])
    self.stage_thresholds = torch.tensor(
      cfg.params["stage_thresholds"], device=env.device, dtype=torch.float32
    )
    self.gate_hold_counter = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self.unlocked = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self.success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    success_values = self.success[env_ids]
    if success_values.numel() > 0:
      self.env.extras["log"]["Episode_Termination/right_object_stage_success"] = (
        torch.count_nonzero(success_values).to(dtype=torch.float32)
        / float(success_values.numel())
      )
    self.gate_hold_counter[env_ids] = 0
    self.unlocked[env_ids] = False
    self.success[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    left_fingertip_cfg: SceneEntityCfg,
    left_target_cfg: SceneEntityCfg,
    right_fingertip_cfg: SceneEntityCfg,
    right_target_cfg: SceneEntityCfg,
    left_sensor_name: str,
    right_sensor_name: str,
    moving_site_cfg: SceneEntityCfg,
    stage_target_cfg: SceneEntityCfg,
    activation_threshold: float,
    activation_force_threshold: float,
    activation_hold_steps: int,
    stage_thresholds: tuple[float, ...],
    stage_exp_scales: tuple[float, ...],
    stage_interp_counts: tuple[int, ...] | None = None,
  ) -> torch.Tensor:
    del (
      activation_hold_steps,
      stage_thresholds,
      stage_exp_scales,
      stage_interp_counts,
      left_fingertip_cfg,
      left_target_cfg,
      left_sensor_name,
    )

    gate_mask = _right_hand_grasp_gate_mask(
      env,
      right_fingertip_cfg=right_fingertip_cfg,
      right_target_cfg=right_target_cfg,
      activation_threshold=activation_threshold,
      right_sensor_name=right_sensor_name,
      activation_force_threshold=activation_force_threshold,
    )
    self.gate_hold_counter, self.unlocked = _update_gate_state(
      self.gate_hold_counter,
      self.unlocked,
      gate_mask,
      self.activation_hold_steps,
    )

    stage_distances = _right_object_stage_distances(
      env,
      moving_site_cfg=moving_site_cfg,
      target_site_cfg=stage_target_cfg,
    )
    target3_distance = stage_distances[:, -1]
    self.success = self.unlocked & (target3_distance < self.stage_thresholds[-1])
    return self.success


class right_object_stage_no_progress:
  """Terminate when unlocked stage progress stalls for too many steps."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    self.activation_hold_steps = int(cfg.params["activation_hold_steps"])
    self.stage_thresholds = torch.tensor(
      cfg.params["stage_thresholds"], device=env.device, dtype=torch.float32
    )
    self.stage_interp_counts = tuple(
      int(count) for count in cfg.params.get("stage_interp_counts", ())
    )
    if not self.stage_interp_counts:
      self.stage_interp_counts = tuple(0 for _ in range(self.stage_thresholds.numel()))
    self.waypoint_thresholds = _repeat_stage_values_by_interp(
      self.stage_thresholds,
      self.stage_interp_counts,
    )
    self.waypoint_count = int(self.waypoint_thresholds.numel())
    self.no_progress_steps = int(cfg.params.get("no_progress_steps", 100))
    self.gate_hold_counter = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self.unlocked = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self.waypoint_index = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self.stalled_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self.has_begin_pos = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self.begin_pos = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.gate_hold_counter[env_ids] = 0
    self.unlocked[env_ids] = False
    self.waypoint_index[env_ids] = 0
    self.stalled_counter[env_ids] = 0
    self.has_begin_pos[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    left_fingertip_cfg: SceneEntityCfg,
    left_target_cfg: SceneEntityCfg,
    right_fingertip_cfg: SceneEntityCfg,
    right_target_cfg: SceneEntityCfg,
    left_sensor_name: str,
    right_sensor_name: str,
    moving_site_cfg: SceneEntityCfg,
    stage_target_cfg: SceneEntityCfg,
    activation_threshold: float,
    activation_force_threshold: float,
    activation_hold_steps: int,
    stage_thresholds: tuple[float, ...],
    stage_exp_scales: tuple[float, ...],
    stage_interp_counts: tuple[int, ...] | None = None,
    no_progress_steps: int | None = None,
  ) -> torch.Tensor:
    del (
      activation_hold_steps,
      stage_thresholds,
      stage_exp_scales,
      stage_interp_counts,
      no_progress_steps,
      left_fingertip_cfg,
      left_target_cfg,
      left_sensor_name,
    )

    gate_mask = _right_hand_grasp_gate_mask(
      env,
      right_fingertip_cfg=right_fingertip_cfg,
      right_target_cfg=right_target_cfg,
      activation_threshold=activation_threshold,
      right_sensor_name=right_sensor_name,
      activation_force_threshold=activation_force_threshold,
    )
    self.gate_hold_counter, self.unlocked = _update_gate_state(
      self.gate_hold_counter,
      self.unlocked,
      gate_mask,
      self.activation_hold_steps,
    )

    moving_pos, stage_target_pos = _right_object_stage_positions(
      env,
      moving_site_cfg=moving_site_cfg,
      target_site_cfg=stage_target_cfg,
    )
    missing_begin = self.unlocked & ~self.has_begin_pos
    self.begin_pos = torch.where(
      missing_begin.unsqueeze(-1),
      moving_pos,
      self.begin_pos,
    )
    self.has_begin_pos = self.has_begin_pos | missing_begin

    waypoints = _right_object_stage_waypoints(
      self.begin_pos,
      stage_target_pos,
      self.stage_interp_counts,
    )
    prev_waypoint_index = self.waypoint_index
    active_waypoint = _active_waypoint_position(
      moving_pos=moving_pos,
      waypoints=waypoints,
      waypoint_index=self.waypoint_index,
    )
    active_distance = torch.linalg.norm(moving_pos - active_waypoint, dim=-1)
    active_index = self.waypoint_index.clamp(max=self.waypoint_count - 1)
    active_threshold = self.waypoint_thresholds.gather(0, active_index)
    reached = (
      self.unlocked
      & (self.waypoint_index < self.waypoint_count)
      & (active_distance < active_threshold)
    )
    self.waypoint_index = torch.where(
      reached,
      self.waypoint_index + 1,
      self.waypoint_index,
    )

    progressed = self.waypoint_index > prev_waypoint_index
    active = self.unlocked & (self.waypoint_index < self.waypoint_count)
    self.stalled_counter = torch.where(
      ~active,
      torch.zeros_like(self.stalled_counter),
      torch.where(
        progressed,
        torch.zeros_like(self.stalled_counter),
        self.stalled_counter + 1,
      ),
    )
    return active & (self.stalled_counter >= self.no_progress_steps)
