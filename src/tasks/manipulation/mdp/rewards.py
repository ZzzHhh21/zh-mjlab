from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class _ObjectSdfGrid:
  def __init__(self, sdf_file: str, device: str):
    data = np.load(sdf_file)
    self.sdf = torch.as_tensor(
      data["sdf"], dtype=torch.float32, device=device
    ).contiguous()
    self.flat_sdf = self.sdf.reshape(-1)
    self.bbox_min = torch.as_tensor(
      data["bbox_min"], dtype=torch.float32, device=device
    )
    self.voxel_size = torch.as_tensor(
      data["voxel_size"], dtype=torch.float32, device=device
    )
    self.grid_shape = torch.tensor(self.sdf.shape, dtype=torch.long, device=device)

  def query(self, points_obj: torch.Tensor) -> torch.Tensor:
    original_shape = points_obj.shape[:-1]
    points = points_obj.reshape(-1, 3)
    coord = (points - self.bbox_min) / self.voxel_size
    max_coord = self.grid_shape.to(torch.float32) - 1.0
    inside_grid = ((coord >= 0.0) & (coord <= max_coord)).all(dim=-1)
    coord = torch.minimum(torch.maximum(coord, torch.zeros_like(coord)), max_coord)

    c0 = torch.floor(coord).to(torch.long)
    c1 = torch.minimum(c0 + 1, self.grid_shape - 1)
    w = coord - c0.to(torch.float32)

    nx, ny, nz = self.grid_shape.unbind()

    def gather(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor) -> torch.Tensor:
      flat_index = ix * ny * nz + iy * nz + iz
      return self.flat_sdf[flat_index]

    x0, y0, z0 = c0[:, 0], c0[:, 1], c0[:, 2]
    x1, y1, z1 = c1[:, 0], c1[:, 1], c1[:, 2]
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]

    c000 = gather(x0, y0, z0)
    c001 = gather(x0, y0, z1)
    c010 = gather(x0, y1, z0)
    c011 = gather(x0, y1, z1)
    c100 = gather(x1, y0, z0)
    c101 = gather(x1, y0, z1)
    c110 = gather(x1, y1, z0)
    c111 = gather(x1, y1, z1)

    c00 = c000 * (1.0 - wx) + c100 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx
    c0v = c00 * (1.0 - wy) + c10 * wy
    c1v = c01 * (1.0 - wy) + c11 * wy
    sdf = c0v * (1.0 - wz) + c1v * wz
    sdf = torch.where(inside_grid, sdf, torch.full_like(sdf, 1.0))
    return sdf.reshape(original_shape)


def _pairwise_site_distances(
  env: ManagerBasedRlEnv,
  source_cfg: SceneEntityCfg,
  target_cfg: SceneEntityCfg,
) -> torch.Tensor:
  source: Entity = env.scene[source_cfg.name]
  target: Entity = env.scene[target_cfg.name]
  source_pos = source.data.site_pos_w[:, source_cfg.site_ids]
  target_pos = target.data.site_pos_w[:, target_cfg.site_ids]
  if source_pos.shape[1] != target_pos.shape[1]:
    raise ValueError("Source/target site count mismatch.")
  return torch.linalg.norm(source_pos - target_pos, dim=-1)


def _grasp_gate_mask(
  env: ManagerBasedRlEnv,
  left_fingertip_cfg: SceneEntityCfg,
  left_target_cfg: SceneEntityCfg,
  right_fingertip_cfg: SceneEntityCfg,
  right_target_cfg: SceneEntityCfg,
  activation_threshold: float,
  left_sensor_name: str,
  right_sensor_name: str,
  activation_force_threshold: float,
) -> torch.Tensor:
  left_dist = _pairwise_site_distances(env, left_fingertip_cfg, left_target_cfg)
  right_dist = _pairwise_site_distances(env, right_fingertip_cfg, right_target_cfg)

  left_sensor: ContactSensor = env.scene[left_sensor_name]
  right_sensor: ContactSensor = env.scene[right_sensor_name]
  left_force = left_sensor.data.force
  right_force = right_sensor.data.force
  if left_force is None or right_force is None:
    raise ValueError("Stage activation requires fingertip-object contact force data.")

  left_force_mag = _contact_force_magnitude(left_force)
  right_force_mag = _contact_force_magnitude(right_force)
  left_required_count = left_dist.shape[1]
  right_required_count = right_dist.shape[1]
  if left_force_mag.shape[1] < left_required_count:
    raise ValueError("Left stage activation contact force count mismatch.")
  if right_force_mag.shape[1] < right_required_count:
    raise ValueError("Right stage activation contact force count mismatch.")

  left_distance_ok = (left_dist < activation_threshold).all(dim=-1)
  right_distance_ok = (right_dist < activation_threshold).all(dim=-1)
  left_force_ok = (
    left_force_mag[:, :left_required_count] > activation_force_threshold
  ).all(dim=-1)
  right_force_ok = (
    right_force_mag[:, :right_required_count] > activation_force_threshold
  ).all(dim=-1)
  return left_distance_ok & right_distance_ok & left_force_ok & right_force_ok


def _right_hand_grasp_gate_mask(
  env: ManagerBasedRlEnv,
  right_fingertip_cfg: SceneEntityCfg,
  right_target_cfg: SceneEntityCfg,
  activation_threshold: float,
  right_sensor_name: str,
  activation_force_threshold: float,
) -> torch.Tensor:
  right_dist = _pairwise_site_distances(env, right_fingertip_cfg, right_target_cfg)
  right_sensor: ContactSensor = env.scene[right_sensor_name]
  right_force = right_sensor.data.force
  if right_force is None:
    raise ValueError("Stage activation requires right fingertip-object force data.")

  right_force_mag = _contact_force_magnitude(right_force)
  required_count = right_dist.shape[1]
  if right_force_mag.shape[1] < required_count:
    raise ValueError("Right stage activation contact force count mismatch.")

  distance_ok = (right_dist < activation_threshold).all(dim=-1)
  force_ok = (right_force_mag[:, :required_count] > activation_force_threshold).all(
    dim=-1
  )
  return distance_ok & force_ok


def _single_hand_grasp_gate_mask(
  env: ManagerBasedRlEnv,
  fingertip_cfg: SceneEntityCfg,
  target_cfg: SceneEntityCfg,
  activation_threshold: float,
  sensor_name: str,
  activation_force_threshold: float,
) -> torch.Tensor:
  distance = _pairwise_site_distances(env, fingertip_cfg, target_cfg)
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  if force is None:
    raise ValueError("Stage activation requires fingertip-object force data.")

  force_mag = _contact_force_magnitude(force)
  required_count = distance.shape[1]
  if force_mag.shape[1] < required_count:
    raise ValueError("Stage activation contact force count mismatch.")

  distance_ok = (distance < activation_threshold).all(dim=-1)
  force_ok = (force_mag[:, :required_count] > activation_force_threshold).all(
    dim=-1
  )
  return distance_ok & force_ok


def _right_object_stage_distances(
  env: ManagerBasedRlEnv,
  moving_site_cfg: SceneEntityCfg,
  target_site_cfg: SceneEntityCfg,
) -> torch.Tensor:
  moving_pos, target_pos = _right_object_stage_positions(
    env,
    moving_site_cfg=moving_site_cfg,
    target_site_cfg=target_site_cfg,
  )
  return torch.linalg.norm(moving_pos[:, None, :] - target_pos, dim=-1)


def _right_object_stage_positions(
  env: ManagerBasedRlEnv,
  moving_site_cfg: SceneEntityCfg,
  target_site_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
  moving_obj: Entity = env.scene[moving_site_cfg.name]
  target_obj: Entity = env.scene[target_site_cfg.name]
  moving_pos = moving_obj.data.site_pos_w[:, moving_site_cfg.site_ids]
  target_pos = target_obj.data.site_pos_w[:, target_site_cfg.site_ids]
  if moving_pos.shape[1] != 1:
    raise ValueError("Moving-site cfg must contain exactly one site.")
  return moving_pos.squeeze(1), target_pos


def _repeat_stage_values_by_interp(
  stage_values: torch.Tensor,
  interp_counts: tuple[int, ...],
) -> torch.Tensor:
  if len(interp_counts) != stage_values.numel():
    raise ValueError("stage_interp_counts length must match stage count.")
  expanded: list[torch.Tensor] = []
  for stage_id, interp_count in enumerate(interp_counts):
    expanded.extend([stage_values[stage_id]] * (int(interp_count) + 1))
  return torch.stack(expanded)


def _right_object_stage_waypoints(
  begin_pos: torch.Tensor,
  stage_target_pos: torch.Tensor,
  interp_counts: tuple[int, ...],
) -> torch.Tensor:
  if stage_target_pos.shape[1] != len(interp_counts):
    raise ValueError("stage_interp_counts length must match target count.")

  waypoints: list[torch.Tensor] = []
  segment_start = begin_pos
  for stage_id, interp_count in enumerate(interp_counts):
    segment_end = stage_target_pos[:, stage_id]
    divisor = float(int(interp_count) + 1)
    for interp_id in range(int(interp_count) + 1):
      alpha = float(interp_id + 1) / divisor
      waypoints.append(segment_start + alpha * (segment_end - segment_start))
    segment_start = segment_end
  return torch.stack(waypoints, dim=1)


def _active_waypoint_position(
  moving_pos: torch.Tensor,
  waypoints: torch.Tensor,
  waypoint_index: torch.Tensor,
) -> torch.Tensor:
  active_index = waypoint_index.clamp(max=waypoints.shape[1] - 1)
  batch_index = torch.arange(moving_pos.shape[0], device=moving_pos.device)
  return waypoints[batch_index, active_index]


def _active_waypoint_segment_progress(
  moving_pos: torch.Tensor,
  begin_pos: torch.Tensor,
  waypoints: torch.Tensor,
  waypoint_index: torch.Tensor,
  path_exp_scale: torch.Tensor,
) -> torch.Tensor:
  segment_start, segment_end = _active_waypoint_segment(
    begin_pos=begin_pos,
    waypoints=waypoints,
    waypoint_index=waypoint_index,
  )
  segment = segment_end - segment_start
  segment_len_sq = torch.sum(torch.square(segment), dim=-1).clamp_min(1e-8)

  raw_progress = torch.sum((moving_pos - segment_start) * segment, dim=-1) / segment_len_sq
  progress = raw_progress.clamp(min=0.0, max=1.0)
  projected = segment_start + progress.unsqueeze(-1) * segment
  lateral_distance = torch.linalg.norm(moving_pos - projected, dim=-1)
  path_alignment = torch.exp(-path_exp_scale * lateral_distance)
  return progress * path_alignment


def _active_waypoint_height_progress(
  moving_pos: torch.Tensor,
  begin_pos: torch.Tensor,
  waypoints: torch.Tensor,
  waypoint_index: torch.Tensor,
) -> torch.Tensor:
  segment_start, segment_end = _active_waypoint_segment(
    begin_pos=begin_pos,
    waypoints=waypoints,
    waypoint_index=waypoint_index,
  )
  z_delta = segment_end[:, 2] - segment_start[:, 2]
  z_abs_delta = torch.abs(z_delta)
  z_progress = (moving_pos[:, 2] - segment_start[:, 2]) / z_delta.clamp(
    min=1e-6
  )
  z_progress = torch.where(
    z_delta < 0.0,
    (segment_start[:, 2] - moving_pos[:, 2]) / (-z_delta).clamp(min=1e-6),
    z_progress,
  )
  return torch.where(
    z_abs_delta > 1e-5,
    z_progress.clamp(min=0.0, max=1.0),
    torch.ones_like(z_abs_delta),
  )


def _active_waypoint_segment(
  begin_pos: torch.Tensor,
  waypoints: torch.Tensor,
  waypoint_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  active_index = waypoint_index.clamp(max=waypoints.shape[1] - 1)
  prev_index = (active_index - 1).clamp(min=0)
  batch_index = torch.arange(begin_pos.shape[0], device=begin_pos.device)

  segment_start = waypoints[batch_index, prev_index]
  segment_start = torch.where(
    (waypoint_index <= 0).unsqueeze(-1),
    begin_pos,
    segment_start,
  )
  segment_end = waypoints[batch_index, active_index]
  return segment_start, segment_end


def _update_gate_state(
  hold_counter: torch.Tensor,
  unlocked: torch.Tensor,
  gate_mask: torch.Tensor,
  hold_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  next_counter = torch.where(
    unlocked,
    hold_counter,
    torch.where(gate_mask, hold_counter + 1, torch.zeros_like(hold_counter)),
  )
  next_unlocked = unlocked | (next_counter >= hold_steps)
  return next_counter, next_unlocked


def _advance_stage_index(
  unlocked: torch.Tensor,
  stage_index: torch.Tensor,
  stage_distances: torch.Tensor,
  stage_thresholds: torch.Tensor,
) -> torch.Tensor:
  next_stage_index = stage_index.clone()
  for stage_id in range(stage_thresholds.numel()):
    reached = (
      unlocked
      & (next_stage_index == stage_id)
      & (stage_distances[:, stage_id] < stage_thresholds[stage_id])
    )
    next_stage_index = torch.where(reached, next_stage_index + 1, next_stage_index)
  return next_stage_index.clamp_max(stage_thresholds.numel())


class right_object_stage_reward:
  """Additive staged reward with ordered interpolated waypoints on the object path."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.env = env
    self.term_weight = float(cfg.weight)
    self.activation_hold_steps = int(cfg.params["activation_hold_steps"])
    self.stage_thresholds = torch.tensor(
      cfg.params["stage_thresholds"], device=env.device, dtype=torch.float32
    )
    self.stage_exp_scales = torch.tensor(
      cfg.params["stage_exp_scales"], device=env.device, dtype=torch.float32
    )
    self.segment_exp_scale = torch.tensor(
      float(cfg.params.get("segment_exp_scale", cfg.params["stage_exp_scales"][0])),
      device=env.device,
      dtype=torch.float32,
    )
    self.height_exp_scale = torch.tensor(
      float(cfg.params.get("height_exp_scale", 30.0)),
      device=env.device,
      dtype=torch.float32,
    )
    self.height_progress_mix = float(cfg.params.get("height_progress_mix", 0.5))
    self.height_track_mix = float(cfg.params.get("height_track_mix", 0.5))
    self.segment_reward_weight = float(cfg.params.get("segment_reward_weight", 0.6))
    self.waypoint_reward_weight = float(cfg.params.get("waypoint_reward_weight", 0.2))
    self.height_reward_weight = float(cfg.params.get("height_reward_weight", 0.2))
    self.passed_waypoint_reward_weight = float(
      cfg.params.get("passed_waypoint_reward_weight", 3.0)
    )
    self.success_reward_multiplier = float(
      cfg.params.get("success_reward_multiplier", 1.0)
    )
    self.stage_reward_max = max(
      self.success_reward_multiplier
      * (
        self.segment_reward_weight
        + self.waypoint_reward_weight
        + self.height_reward_weight
      ),
      1e-6,
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
    self.waypoint_exp_scales = _repeat_stage_values_by_interp(
      self.stage_exp_scales,
      self.stage_interp_counts,
    )
    self.waypoint_count = int(self.waypoint_thresholds.numel())
    self.gate_hold_counter = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self.unlocked = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self.waypoint_index = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self.episode_waypoint_max = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self.has_begin_pos = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self.begin_pos = torch.zeros(
      env.num_envs, 3, device=env.device, dtype=torch.float32
    )
    self.episode_mins = _make_episode_min_buffers(
      env,
      (
        "right_obj_target1_distance",
        "right_obj_target2_distance",
        "right_obj_target3_distance",
      ),
    )
    self.episode_sums = _make_episode_sum_buffers(
      env,
      (
        "right_object_stage_segment_progress",
        "right_object_stage_waypoint_exp",
        "right_object_stage_height_reward",
      ),
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    _flush_episode_min_buffers(self.env, self.episode_mins, env_ids)
    _flush_episode_sum_buffers(self.env, self.episode_sums, env_ids)
    episode_waypoint_max = torch.max(self.episode_waypoint_max[env_ids])
    self.env.extras["log"]["Episode_Reward/right_object_stage_max_waypoint_index"] = (
      episode_waypoint_max
    )
    self.episode_waypoint_max[env_ids] = 0.0
    self.gate_hold_counter[env_ids] = 0
    self.unlocked[env_ids] = False
    self.waypoint_index[env_ids] = 0
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
    segment_exp_scale: float,
    stage_interp_counts: tuple[int, ...],
    height_exp_scale: float | None = None,
    height_progress_mix: float | None = None,
    height_track_mix: float | None = None,
    segment_reward_weight: float | None = None,
    waypoint_reward_weight: float | None = None,
    height_reward_weight: float | None = None,
    passed_waypoint_reward_weight: float | None = None,
    success_reward_multiplier: float | None = None,
  ) -> torch.Tensor:
    del (
      activation_hold_steps,
      stage_thresholds,
      stage_exp_scales,
      segment_exp_scale,
      stage_interp_counts,
      height_exp_scale,
      height_progress_mix,
      height_track_mix,
      segment_reward_weight,
      waypoint_reward_weight,
      height_reward_weight,
      passed_waypoint_reward_weight,
      success_reward_multiplier,
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
    target_delta = moving_pos[:, None, :] - stage_target_pos
    target_distance = torch.linalg.norm(target_delta, dim=-1)
    final_target_success = self.unlocked & (
      target_distance[:, -1] < self.stage_thresholds[-1]
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
    self.episode_waypoint_max = torch.maximum(
      self.episode_waypoint_max,
      torch.where(
        final_target_success,
        torch.full_like(
          self.waypoint_index,
          self.waypoint_count,
        ),
        self.waypoint_index,
      ).to(dtype=torch.float32),
    )

    reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    segment_progress_reward = torch.zeros_like(reward)
    waypoint_reward = torch.zeros_like(reward)
    height_exp_reward = torch.zeros_like(reward)
    height_reward = torch.zeros_like(reward)
    height_progress_reward = torch.zeros_like(reward)
    active_mask = self.unlocked & (self.waypoint_index < self.waypoint_count)
    active_index = self.waypoint_index.clamp(max=self.waypoint_count - 1)
    active_waypoint = _active_waypoint_position(
      moving_pos=moving_pos,
      waypoints=waypoints,
      waypoint_index=self.waypoint_index,
    )
    active_distance = torch.linalg.norm(moving_pos - active_waypoint, dim=-1)
    active_scale = self.waypoint_exp_scales.gather(0, active_index)
    waypoint_reward = torch.exp(-active_scale * active_distance)
    height_error = torch.abs(moving_pos[:, 2] - active_waypoint[:, 2])
    height_exp_reward = torch.exp(-self.height_exp_scale * height_error)
    height_progress_reward = _active_waypoint_height_progress(
      moving_pos=moving_pos,
      begin_pos=self.begin_pos,
      waypoints=waypoints,
      waypoint_index=self.waypoint_index,
    )
    height_reward = height_exp_reward
    segment_progress_reward = _active_waypoint_segment_progress(
      moving_pos=moving_pos,
      begin_pos=self.begin_pos,
      waypoints=waypoints,
      waypoint_index=self.waypoint_index,
      path_exp_scale=self.segment_exp_scale.expand_as(active_scale),
    )
    segment_progress_reward = torch.where(
      active_mask,
      segment_progress_reward,
      torch.zeros_like(segment_progress_reward),
    )
    waypoint_reward = torch.where(
      active_mask,
      waypoint_reward,
      torch.zeros_like(waypoint_reward),
    )
    height_reward = torch.where(
      active_mask,
      height_reward,
      torch.zeros_like(height_reward),
    )
    height_exp_reward = torch.where(
      active_mask,
      height_exp_reward,
      torch.zeros_like(height_exp_reward),
    )
    height_progress_reward = torch.where(
      active_mask,
      height_progress_reward,
      torch.zeros_like(height_progress_reward),
    )
    dense_reward = (
      self.segment_reward_weight * segment_progress_reward
      + self.waypoint_reward_weight * waypoint_reward
      + self.height_reward_weight * height_reward
    )
    progress_bonus = (
      self.passed_waypoint_reward_weight
      * self.waypoint_index.to(dtype=torch.float32)
    )
    reward = torch.where(
      active_mask,
      progress_bonus + dense_reward / float(self.waypoint_count),
      reward,
    )

    success_mask = final_target_success
    reward = torch.where(
      success_mask,
      torch.full_like(reward, self.stage_reward_max),
      reward,
    )

    step_scale = self.term_weight * env.step_dt

    self.episode_mins["right_obj_target1_distance"] = torch.minimum(
      self.episode_mins["right_obj_target1_distance"], target_distance[:, 0]
    )
    self.episode_mins["right_obj_target2_distance"] = torch.minimum(
      self.episode_mins["right_obj_target2_distance"], target_distance[:, 1]
    )
    self.episode_mins["right_obj_target3_distance"] = torch.minimum(
      self.episode_mins["right_obj_target3_distance"], target_distance[:, 2]
    )
    self.episode_sums["right_object_stage_segment_progress"] += (
      step_scale * segment_progress_reward
    )
    self.episode_sums["right_object_stage_waypoint_exp"] += (
      step_scale * waypoint_reward
    )
    self.episode_sums["right_object_stage_height_reward"] += (
      step_scale * height_reward
    )

    return reward


class contact_move_object_regularization_reward:
  """Object regularization for contact-first residual training."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.env = env
    self.term_weight = float(cfg.weight)
    self.activation_hold_steps = int(cfg.params["activation_hold_steps"])
    self.include_left = bool(cfg.params.get("include_left", True))
    self.include_right = bool(cfg.params.get("include_right", True))
    self.gate_hold_counter = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self.unlocked = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self.initialized = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self.left_initial_pos = torch.zeros(env.num_envs, 3, device=env.device)
    self.right_initial_pos = torch.zeros(env.num_envs, 3, device=env.device)
    self.allow_right_object_motion_after_grasp = bool(
      cfg.params.get("allow_right_object_motion_after_grasp", False)
    )
    episode_keys: list[str] = []
    if self.include_left:
      episode_keys.append("contact_move_left_object_stability")
    if self.include_right:
      if self.allow_right_object_motion_after_grasp:
        episode_keys.append("contact_move_right_speed")
      else:
        episode_keys.extend(
          (
            "contact_move_right_object_stability",
            "contact_move_right_lift",
          )
        )
    self.episode_sums = _make_episode_sum_buffers(
      env,
      tuple(episode_keys),
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    _flush_episode_sum_buffers(self.env, self.episode_sums, env_ids)
    self.gate_hold_counter[env_ids] = 0
    self.unlocked[env_ids] = False
    self.initialized[env_ids] = False
    self.left_initial_pos[env_ids] = 0.0
    self.right_initial_pos[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    left_fingertip_cfg: SceneEntityCfg,
    left_target_cfg: SceneEntityCfg,
    right_fingertip_cfg: SceneEntityCfg,
    right_target_cfg: SceneEntityCfg,
    left_sensor_name: str,
    right_sensor_name: str,
    activation_threshold: float,
    activation_force_threshold: float,
    activation_hold_steps: int,
    allow_right_object_motion_after_grasp: bool,
    left_object_name: str,
    right_object_name: str,
    lift_tolerance: float,
    linear_velocity_tolerance: float,
    angular_velocity_tolerance: float,
    left_xy_displacement_tolerance: float,
    right_xy_displacement_tolerance: float,
    left_linear_velocity_weight: float,
    left_angular_velocity_weight: float,
    left_xy_displacement_weight: float,
    left_lift_weight: float,
    right_pre_grasp_linear_velocity_weight: float,
    right_pre_grasp_angular_velocity_weight: float,
    right_xy_displacement_weight: float,
    right_move_linear_velocity_limit: float,
    right_move_angular_velocity_limit: float,
    right_move_linear_velocity_weight: float,
    right_move_angular_velocity_weight: float,
    right_speed_weight: float,
    right_lift_weight: float,
    include_left: bool = True,
    include_right: bool = True,
  ) -> torch.Tensor:
    del activation_hold_steps
    include_left = self.include_left and bool(include_left)
    include_right = self.include_right and bool(include_right)

    left_obj: Entity | None = env.scene[left_object_name] if include_left else None
    right_obj: Entity | None = env.scene[right_object_name] if include_right else None

    uninitialized = ~self.initialized
    if uninitialized.any():
      if left_obj is not None:
        self.left_initial_pos[uninitialized] = left_obj.data.root_link_pos_w[
          uninitialized
        ]
      if right_obj is not None:
        self.right_initial_pos[uninitialized] = right_obj.data.root_link_pos_w[
          uninitialized
        ]
      self.initialized[uninitialized] = True

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

    if left_obj is not None:
      left_lin_speed = torch.linalg.norm(left_obj.data.root_link_lin_vel_w, dim=-1)
      left_ang_speed = torch.linalg.norm(left_obj.data.root_link_ang_vel_w, dim=-1)
      left_lin_cost = torch.square(
        (left_lin_speed - float(linear_velocity_tolerance)).clamp_min(0.0)
      )
      left_ang_cost = torch.square(
        (left_ang_speed - float(angular_velocity_tolerance)).clamp_min(0.0)
      )
      left_xy_disp = torch.linalg.norm(
        left_obj.data.root_link_pos_w[:, :2] - self.left_initial_pos[:, :2],
        dim=-1,
      )
      left_xy_cost = (
        left_xy_disp - float(left_xy_displacement_tolerance)
      ).clamp_min(0.0)
      left_lift = (
        left_obj.data.root_link_pos_w[:, 2]
        - self.left_initial_pos[:, 2]
        - float(lift_tolerance)
      ).clamp_min(0.0)
      left_term = (
        left_linear_velocity_weight * left_lin_cost
        + left_angular_velocity_weight * left_ang_cost
        + left_xy_displacement_weight * left_xy_cost
        + left_lift_weight * left_lift
      )
    else:
      left_term = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    if right_obj is not None:
      right_lin_speed = torch.linalg.norm(right_obj.data.root_link_lin_vel_w, dim=-1)
      right_ang_speed = torch.linalg.norm(right_obj.data.root_link_ang_vel_w, dim=-1)
      right_lin_cost = torch.square(
        (right_lin_speed - float(linear_velocity_tolerance)).clamp_min(0.0)
      )
      right_ang_cost = torch.square(
        (right_ang_speed - float(angular_velocity_tolerance)).clamp_min(0.0)
      )
      right_xy_disp = torch.linalg.norm(
        right_obj.data.root_link_pos_w[:, :2] - self.right_initial_pos[:, :2],
        dim=-1,
      )
      right_xy_cost = (
        right_xy_disp - float(right_xy_displacement_tolerance)
      ).clamp_min(0.0)
      right_lift = (
        right_obj.data.root_link_pos_w[:, 2]
        - self.right_initial_pos[:, 2]
        - float(lift_tolerance)
      ).clamp_min(0.0)
      if allow_right_object_motion_after_grasp:
        right_object_stability_term = torch.zeros_like(right_lin_cost)
        right_lift_term = torch.zeros_like(right_lift)
        right_move_lin_cost = torch.square(
          (right_lin_speed - float(right_move_linear_velocity_limit)).clamp_min(0.0)
        )
        right_move_ang_cost = torch.square(
          (right_ang_speed - float(right_move_angular_velocity_limit)).clamp_min(0.0)
        )
        right_speed_term = self.unlocked.to(dtype=torch.float32) * (
          float(right_speed_weight)
          * (
            right_move_linear_velocity_weight * right_move_lin_cost
            + right_move_angular_velocity_weight * right_move_ang_cost
          )
        )
      else:
        right_object_stability_term = (
          right_pre_grasp_linear_velocity_weight * right_lin_cost
          + right_pre_grasp_angular_velocity_weight * right_ang_cost
          + right_xy_displacement_weight * right_xy_cost
        )
        right_speed_term = torch.zeros_like(right_lin_cost)
        right_lift_term = right_lift_weight * right_lift
    else:
      right_object_stability_term = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.float32
      )
      right_speed_term = torch.zeros_like(right_object_stability_term)
      right_lift_term = torch.zeros_like(right_object_stability_term)

    step_scale = self.term_weight * env.step_dt
    if "contact_move_left_object_stability" in self.episode_sums:
      self.episode_sums["contact_move_left_object_stability"] += step_scale * left_term
    if self.allow_right_object_motion_after_grasp:
      if "contact_move_right_speed" in self.episode_sums:
        self.episode_sums["contact_move_right_speed"] += step_scale * right_speed_term
    else:
      if "contact_move_right_object_stability" in self.episode_sums:
        self.episode_sums["contact_move_right_object_stability"] += (
          step_scale * right_object_stability_term
        )
      if "contact_move_right_lift" in self.episode_sums:
        self.episode_sums["contact_move_right_lift"] += step_scale * right_lift_term

    return left_term + right_object_stability_term + right_speed_term + right_lift_term


class object_reset_rotation_penalty:
  """Penalize object root rotation relative to the pose captured at reset."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.env = env
    self.object_name = str(cfg.params["object_name"])
    self.reference_quat = torch.zeros(
      env.num_envs, 4, device=env.device, dtype=torch.float32
    )
    self.has_reference = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def _current_quat(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    obj: Entity = env.scene[self.object_name]
    free_q_adr = obj.data.indexing.free_joint_q_adr
    if free_q_adr.numel() < 7:
      raise ValueError(f"Object '{self.object_name}' must have a free joint.")
    return obj.data.data.qpos[:, free_q_adr[3:7]]

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    current_quat = self._current_quat(self.env)
    self.reference_quat[env_ids] = current_quat[env_ids]
    self.has_reference[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    rotation_threshold: float,
  ) -> torch.Tensor:
    if object_name != self.object_name:
      raise ValueError(
        f"Configured object '{object_name}' does not match '{self.object_name}'."
      )

    current_quat = self._current_quat(env)
    missing_reference = ~self.has_reference
    if missing_reference.any():
      self.reference_quat[missing_reference] = current_quat[missing_reference]
      self.has_reference[missing_reference] = True

    safe_current_quat = torch.nan_to_num(
      current_quat, nan=0.0, posinf=0.0, neginf=0.0
    )
    angle = quat_error_magnitude(safe_current_quat, self.reference_quat)
    angle = torch.nan_to_num(angle, nan=0.0, posinf=1e6, neginf=0.0)
    return (angle - float(rotation_threshold)).clamp_min(0.0)


def joint_velocity_reward(
  env: ManagerBasedRlEnv,
  scale: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Positive smoothness reward for small joint velocity norm: exp(-k * ||v||)."""
  robot: Entity = env.scene[asset_cfg.name]
  joint_ids = _valid_joint_ids(asset_cfg.joint_ids, robot.data.joint_vel.shape[1])
  if torch.is_tensor(joint_ids) and joint_ids.numel() == 0:
    return torch.ones(env.num_envs, dtype=torch.float32, device=env.device)
  joint_vel = robot.data.joint_vel[:, joint_ids]
  error = torch.linalg.norm(joint_vel, dim=-1)
  return torch.exp(-scale * error)


def joint_acceleration_reward(
  env: ManagerBasedRlEnv,
  scale: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Positive smoothness reward for small joint acceleration norm: exp(-k * ||a||)."""
  robot: Entity = env.scene[asset_cfg.name]
  joint_ids = _valid_joint_ids(asset_cfg.joint_ids, robot.data.joint_acc.shape[1])
  if torch.is_tensor(joint_ids) and joint_ids.numel() == 0:
    return torch.ones(env.num_envs, dtype=torch.float32, device=env.device)
  joint_acc = robot.data.joint_acc[:, joint_ids]
  error = torch.linalg.norm(joint_acc, dim=-1)
  return torch.exp(-scale * error)


def _valid_joint_ids(
  joint_ids,
  joint_count: int,
) -> torch.Tensor | slice:
  if isinstance(joint_ids, slice):
    return joint_ids
  if not torch.is_tensor(joint_ids):
    joint_ids = torch.as_tensor(joint_ids, dtype=torch.long)
  valid_mask = (joint_ids >= 0) & (joint_ids < int(joint_count))
  return joint_ids[valid_mask]


def _joint_pos_limit_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  soft_joint_pos_limits = robot.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None
  joint_count = min(robot.data.joint_pos.shape[1], soft_joint_pos_limits.shape[1])
  joint_ids = _valid_joint_ids(asset_cfg.joint_ids, joint_count)
  if torch.is_tensor(joint_ids) and joint_ids.numel() == 0:
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
  out_of_limits = -(
    robot.data.joint_pos[:, joint_ids]
    - soft_joint_pos_limits[:, joint_ids, 0]
  ).clip(max=0.0)
  out_of_limits += (
    robot.data.joint_pos[:, joint_ids]
    - soft_joint_pos_limits[:, joint_ids, 1]
  ).clip(min=0.0)
  return torch.sum(out_of_limits, dim=1)


def action_rate_reward(
  env: ManagerBasedRlEnv,
  scale: float,
) -> torch.Tensor:
  """Positive smoothness reward for small action-rate norm: exp(-k * ||da||)."""
  action_rate = env.action_manager.action - env.action_manager.prev_action
  error = torch.linalg.norm(action_rate, dim=-1)
  return torch.exp(-scale * error)


def _action_rate_reward_subset(
  env: ManagerBasedRlEnv,
  scale: float,
  action_ids: torch.Tensor,
) -> torch.Tensor:
  action_rate = (
    env.action_manager.action[:, action_ids]
    - env.action_manager.prev_action[:, action_ids]
  )
  error = torch.linalg.norm(action_rate, dim=-1)
  return torch.exp(-scale * error)


def _action_acceleration_reward_subset(
  env: ManagerBasedRlEnv,
  scale: float,
  action_ids: torch.Tensor,
) -> torch.Tensor:
  action_acc = (
    env.action_manager.action[:, action_ids]
    - 2 * env.action_manager.prev_action[:, action_ids]
    + env.action_manager.prev_prev_action[:, action_ids]
  )
  error = torch.sum(torch.square(action_acc), dim=-1)
  return torch.exp(-scale * error)


def _action_magnitude_reward_subset(
  env: ManagerBasedRlEnv,
  scale: float,
  action_ids: torch.Tensor,
) -> torch.Tensor:
  action = env.action_manager.action[:, action_ids]
  error = torch.sum(torch.square(action), dim=-1)
  return torch.exp(-scale * error)


def _contact_force_magnitude(force: torch.Tensor) -> torch.Tensor:
  if force.ndim == 4 and force.shape[-2] == 1:
    force = force.squeeze(-2)
  force_magnitude = torch.linalg.norm(force, dim=-1)
  force_magnitude = torch.nan_to_num(
    force_magnitude, nan=0.0, posinf=0.0, neginf=0.0
  )
  return force_magnitude.clamp_min(0.0)


def _contact_sensor_force_sum(sensor: ContactSensor) -> torch.Tensor:
  data = sensor.data
  if data.force_history is not None:
    force_mag = _contact_force_magnitude(data.force_history)
    return force_mag.amax(dim=-1).sum(dim=-1)

  force = data.force
  if force is None:
    if data.found is None:
      raise ValueError("Contact sensor must provide force or found data.")
    return torch.zeros(data.found.shape[0], dtype=torch.float32, device=data.found.device)

  force_magnitude = _contact_force_magnitude(force)
  return force_magnitude.sum(dim=-1)


def _required_contact_fraction(
  current: torch.Tensor,
  target_graph: torch.Tensor,
) -> torch.Tensor:
  required_mask = (target_graph > 0.5).float()
  required_count = required_mask.sum(dim=-1).clamp_min(1.0)
  contacted_required = (required_mask * current).sum(dim=-1)
  return contacted_required / required_count


def _contact_graph_exp_reward(
  current: torch.Tensor,
  target_graph: torch.Tensor,
  exp_scale: float,
) -> torch.Tensor:
  target = current.new_tensor(target_graph).unsqueeze(0)
  if current.shape[1] != target.shape[1]:
    raise ValueError("Contact graph length must match current contact count.")
  total_error = torch.abs(current - target).sum(dim=-1)
  return torch.exp(-float(exp_scale) * total_error)


def palm_ground_contact_force_penalty(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
) -> torch.Tensor:
  """Penalty proportional to palm-ground contact force magnitude."""
  penalty = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
  for sensor_name in sensor_names:
    sensor: ContactSensor = env.scene[sensor_name]
    penalty += _contact_sensor_force_sum(sensor)
  return penalty


def fixed_contact_graph_alignment_reward(
  env: ManagerBasedRlEnv,
  left_fingertip_cfg: SceneEntityCfg,
  left_target_cfg: SceneEntityCfg,
  right_fingertip_cfg: SceneEntityCfg,
  right_target_cfg: SceneEntityCfg,
  left_sensor_name: str,
  right_sensor_name: str,
  left_target_graph: tuple[float, ...],
  right_target_graph: tuple[float, ...],
  distance_threshold: float,
  force_threshold: float,
) -> torch.Tensor:
  """Normalized reward for how many required fingertips are currently in contact."""
  robot: Entity = env.scene[left_fingertip_cfg.name]
  left_obj: Entity = env.scene[left_target_cfg.name]
  right_obj: Entity = env.scene[right_target_cfg.name]
  left_sensor: ContactSensor = env.scene[left_sensor_name]
  right_sensor: ContactSensor = env.scene[right_sensor_name]

  left_tip_pos = robot.data.site_pos_w[:, left_fingertip_cfg.site_ids]
  left_target_pos = left_obj.data.site_pos_w[:, left_target_cfg.site_ids]
  right_tip_pos = robot.data.site_pos_w[:, right_fingertip_cfg.site_ids]
  right_target_pos = right_obj.data.site_pos_w[:, right_target_cfg.site_ids]

  if left_tip_pos.shape[1] != len(left_target_graph):
    raise ValueError("left_target_graph length must match left site count.")
  if right_tip_pos.shape[1] != len(right_target_graph):
    raise ValueError("right_target_graph length must match right site count.")

  left_force = left_sensor.data.force
  right_force = right_sensor.data.force
  assert left_force is not None
  assert right_force is not None

  left_dist = torch.linalg.norm(left_tip_pos - left_target_pos, dim=-1)
  right_dist = torch.linalg.norm(right_tip_pos - right_target_pos, dim=-1)
  left_force_mag = _contact_force_magnitude(left_force)
  right_force_mag = _contact_force_magnitude(right_force)

  if left_force_mag.shape[1] != left_tip_pos.shape[1]:
    raise ValueError("Left contact force count mismatch.")
  if right_force_mag.shape[1] != right_tip_pos.shape[1]:
    raise ValueError("Right contact force count mismatch.")

  left_current = (
    (left_dist < distance_threshold) & (left_force_mag > force_threshold)
  ).float()
  right_current = (
    (right_dist < distance_threshold) & (right_force_mag > force_threshold)
  ).float()

  left_target = left_current.new_tensor(left_target_graph).unsqueeze(0)
  right_target = right_current.new_tensor(right_target_graph).unsqueeze(0)
  left_reward = _required_contact_fraction(left_current, left_target)
  right_reward = _required_contact_fraction(right_current, right_target)
  return 0.5 * (left_reward + right_reward)


def _single_hand_contact_graph_alignment_reward(
  env: ManagerBasedRlEnv,
  fingertip_cfg: SceneEntityCfg,
  target_cfg: SceneEntityCfg,
  sensor_name: str,
  target_graph: tuple[float, ...],
  distance_threshold: float,
  force_threshold: float,
  exp_scale: float,
) -> torch.Tensor:
  robot: Entity = env.scene[fingertip_cfg.name]
  obj: Entity = env.scene[target_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]

  tip_pos = robot.data.site_pos_w[:, fingertip_cfg.site_ids]
  target_pos = obj.data.site_pos_w[:, target_cfg.site_ids]
  if tip_pos.shape[1] != len(target_graph):
    raise ValueError("target_graph length must match site count.")

  force = sensor.data.force
  assert force is not None
  force_mag = _contact_force_magnitude(force)
  if force_mag.shape[1] != tip_pos.shape[1]:
    raise ValueError("Contact force count mismatch.")

  dist = torch.linalg.norm(tip_pos - target_pos, dim=-1)
  current = ((dist < distance_threshold) & (force_mag > force_threshold)).float()
  return _contact_graph_exp_reward(
    current=current,
    target_graph=target_graph,
    exp_scale=exp_scale,
  )


def single_hand_contact_graph_alignment_reward(
  env: ManagerBasedRlEnv,
  fingertip_cfg: SceneEntityCfg,
  target_cfg: SceneEntityCfg,
  sensor_name: str,
  target_graph: tuple[float, ...],
  distance_threshold: float,
  force_threshold: float,
  exp_scale: float,
) -> torch.Tensor:
  """Single-side exp contact graph reward for separate left/right logging."""
  return _single_hand_contact_graph_alignment_reward(
    env,
    fingertip_cfg=fingertip_cfg,
    target_cfg=target_cfg,
    sensor_name=sensor_name,
    target_graph=target_graph,
    distance_threshold=distance_threshold,
    force_threshold=force_threshold,
    exp_scale=exp_scale,
  )


def single_hand_key_fingertip_force_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_graph: tuple[float, ...],
  force_scale: float,
  eps: float = 1e-5,
) -> torch.Tensor:
  """Dense reward for establishing force on target contact fingertips."""
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  force_mag = _contact_force_magnitude(force)
  if force_mag.shape[1] != len(target_graph):
    raise ValueError("target_graph length must match contact force count.")
  target_mask = force_mag.new_tensor(target_graph).unsqueeze(0)
  key_force_sum = (force_mag * target_mask).sum(dim=-1)
  return torch.exp(-float(force_scale) / (key_force_sum + float(eps)))


def key_fingertip_target_distance_penalty(
  env: ManagerBasedRlEnv,
  left_fingertip_cfg: SceneEntityCfg,
  left_target_cfg: SceneEntityCfg,
  right_fingertip_cfg: SceneEntityCfg,
  right_target_cfg: SceneEntityCfg,
  distance_threshold: float,
  include_left: bool = True,
  include_right: bool = True,
) -> torch.Tensor:
  """Large penalty when residual training moves key fingertips away from targets."""
  penalty = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
  if include_left:
    left_dist = _pairwise_site_distances(env, left_fingertip_cfg, left_target_cfg)
    left_violation = (left_dist - float(distance_threshold)).clamp_min(0.0)
    penalty = penalty + torch.sum(torch.square(left_violation), dim=-1)
  if include_right:
    right_dist = _pairwise_site_distances(env, right_fingertip_cfg, right_target_cfg)
    right_violation = (right_dist - float(distance_threshold)).clamp_min(0.0)
    penalty = penalty + torch.sum(torch.square(right_violation), dim=-1)
  return penalty


def fingertip_wrist_target_alignment_reward(
  env: ManagerBasedRlEnv,
  left_fingertip_cfg: SceneEntityCfg,
  left_target_cfg: SceneEntityCfg,
  right_fingertip_cfg: SceneEntityCfg,
  right_target_cfg: SceneEntityCfg,
  left_wrist_cfg: SceneEntityCfg,
  left_wrist_target_cfg: SceneEntityCfg,
  right_wrist_cfg: SceneEntityCfg,
  right_wrist_target_cfg: SceneEntityCfg,
  left_scales: tuple[float, ...],
  right_scales: tuple[float, ...],
  left_wrist_scale: float,
  right_wrist_scale: float,
) -> torch.Tensor:
  """Per-hand reward with fingertip and wrist multiplicative aggregation."""
  robot: Entity = env.scene[left_fingertip_cfg.name]
  left_obj: Entity = env.scene[left_target_cfg.name]
  right_obj: Entity = env.scene[right_target_cfg.name]

  left_tip_pos = robot.data.site_pos_w[:, left_fingertip_cfg.site_ids]
  left_target_pos = left_obj.data.site_pos_w[:, left_target_cfg.site_ids]
  right_tip_pos = robot.data.site_pos_w[:, right_fingertip_cfg.site_ids]
  right_target_pos = right_obj.data.site_pos_w[:, right_target_cfg.site_ids]
  left_wrist_pos = robot.data.site_pos_w[:, left_wrist_cfg.site_ids]
  left_wrist_target_pos = left_obj.data.site_pos_w[:, left_wrist_target_cfg.site_ids]
  right_wrist_pos = robot.data.site_pos_w[:, right_wrist_cfg.site_ids]
  right_wrist_target_pos = right_obj.data.site_pos_w[:, right_wrist_target_cfg.site_ids]

  if left_tip_pos.shape[1] != left_target_pos.shape[1]:
    raise ValueError("Left fingertip/target site count mismatch.")
  if right_tip_pos.shape[1] != right_target_pos.shape[1]:
    raise ValueError("Right fingertip/target site count mismatch.")
  if left_tip_pos.shape[1] != len(left_scales):
    raise ValueError("left_scales length must match left site count.")
  if right_tip_pos.shape[1] != len(right_scales):
    raise ValueError("right_scales length must match right site count.")
  if left_wrist_pos.shape[1] != 1 or left_wrist_target_pos.shape[1] != 1:
    raise ValueError("Left wrist/target site count mismatch.")
  if right_wrist_pos.shape[1] != 1 or right_wrist_target_pos.shape[1] != 1:
    raise ValueError("Right wrist/target site count mismatch.")

  left_dist = torch.linalg.norm(left_tip_pos - left_target_pos, dim=-1)
  right_dist = torch.linalg.norm(right_tip_pos - right_target_pos, dim=-1)
  left_wrist_dist = torch.linalg.norm(left_wrist_pos - left_wrist_target_pos, dim=-1)
  right_wrist_dist = torch.linalg.norm(right_wrist_pos - right_wrist_target_pos, dim=-1)

  left_scale_tensor = left_dist.new_tensor(left_scales).unsqueeze(0)
  right_scale_tensor = right_dist.new_tensor(right_scales).unsqueeze(0)
  left_reward = torch.exp(-left_scale_tensor * left_dist)
  right_reward = torch.exp(-right_scale_tensor * right_dist)
  left_wrist_reward = torch.exp(-left_wrist_scale * left_wrist_dist).squeeze(-1)
  right_wrist_reward = torch.exp(-right_wrist_scale * right_wrist_dist).squeeze(-1)

  # Match dexhand-style composition: multiply sub-rewards instead of averaging.
  left_reward_prod = torch.prod(left_reward, dim=-1) * left_wrist_reward
  right_reward_prod = torch.prod(right_reward, dim=-1) * right_wrist_reward
  return left_reward_prod + right_reward_prod


def _single_hand_fingertip_wrist_target_alignment_reward(
  env: ManagerBasedRlEnv,
  fingertip_cfg: SceneEntityCfg,
  target_cfg: SceneEntityCfg,
  wrist_cfg: SceneEntityCfg,
  wrist_target_cfg: SceneEntityCfg,
  scales: tuple[float, ...],
  wrist_scale: float,
) -> torch.Tensor:
  robot: Entity = env.scene[fingertip_cfg.name]
  obj: Entity = env.scene[target_cfg.name]

  tip_pos = robot.data.site_pos_w[:, fingertip_cfg.site_ids]
  target_pos = obj.data.site_pos_w[:, target_cfg.site_ids]
  wrist_pos = robot.data.site_pos_w[:, wrist_cfg.site_ids]
  wrist_target_pos = obj.data.site_pos_w[:, wrist_target_cfg.site_ids]

  if tip_pos.shape[1] != target_pos.shape[1]:
    raise ValueError("Fingertip/target site count mismatch.")
  if tip_pos.shape[1] != len(scales):
    raise ValueError("scale length must match site count.")
  if wrist_pos.shape[1] != 1 or wrist_target_pos.shape[1] != 1:
    raise ValueError("Wrist/target site count mismatch.")

  tip_dist = torch.linalg.norm(tip_pos - target_pos, dim=-1)
  wrist_dist = torch.linalg.norm(wrist_pos - wrist_target_pos, dim=-1)

  scale_tensor = tip_dist.new_tensor(scales).unsqueeze(0)
  tip_reward = torch.exp(-scale_tensor * tip_dist)
  wrist_reward = torch.exp(-wrist_scale * wrist_dist).squeeze(-1)
  return torch.prod(tip_reward, dim=-1) * wrist_reward


def _single_hand_joint_site_target_alignment_reward(
  env: ManagerBasedRlEnv,
  joint_site_cfg: SceneEntityCfg,
  joint_target_cfg: SceneEntityCfg,
  scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  robot: Entity = env.scene[joint_site_cfg.name]
  obj: Entity = env.scene[joint_target_cfg.name]

  joint_site_pos = robot.data.site_pos_w[:, joint_site_cfg.site_ids]
  joint_target_pos = obj.data.site_pos_w[:, joint_target_cfg.site_ids]

  if joint_site_pos.shape[1] != joint_target_pos.shape[1]:
    raise ValueError("Joint site/target site count mismatch.")

  joint_dist = torch.linalg.norm(joint_site_pos - joint_target_pos, dim=-1)
  finger_count = 5
  if joint_dist.shape[1] % finger_count != 0:
    raise ValueError("Joint site count must be a multiple of 5 fingers.")

  joint_level_count = joint_dist.shape[1] // finger_count
  primary_group_dist = joint_dist.new_zeros(joint_dist.shape[0])
  secondary_group_dist = joint_dist.new_zeros(joint_dist.shape[0])

  for joint_level in range(joint_level_count):
    level_start = joint_level * finger_count
    primary_group_dist += joint_dist[:, level_start : level_start + 3].sum(dim=-1)
    secondary_group_dist += joint_dist[:, level_start + 3 : level_start + 5].sum(
      dim=-1
    )

  primary_group_reward = torch.exp(-scale * primary_group_dist)
  secondary_group_reward = torch.exp(-scale * secondary_group_dist)
  return primary_group_reward, secondary_group_reward


def _single_hand_primary_site_height_penalty(
  env: ManagerBasedRlEnv,
  fingertip_cfg: SceneEntityCfg,
  target_cfg: SceneEntityCfg,
  joint_site_cfg: SceneEntityCfg,
  joint_target_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot: Entity = env.scene[fingertip_cfg.name]
  obj: Entity = env.scene[target_cfg.name]

  tip_z = robot.data.site_pos_w[:, fingertip_cfg.site_ids, 2]
  target_z = obj.data.site_pos_w[:, target_cfg.site_ids, 2]
  joint_site_z = robot.data.site_pos_w[:, joint_site_cfg.site_ids, 2]
  joint_target_z = obj.data.site_pos_w[:, joint_target_cfg.site_ids, 2]

  if tip_z.shape[1] != target_z.shape[1]:
    raise ValueError("Fingertip/target site count mismatch.")
  if joint_site_z.shape[1] != joint_target_z.shape[1]:
    raise ValueError("Joint site/target site count mismatch.")

  finger_count = 5
  primary_finger_count = 3
  if tip_z.shape[1] < primary_finger_count:
    raise ValueError("Fingertip site count must include thumb/index/middle.")
  if joint_site_z.shape[1] % finger_count != 0:
    raise ValueError("Joint site count must be a multiple of 5 fingers.")

  violation = (target_z[:, :primary_finger_count] - tip_z[:, :primary_finger_count]).clamp_min(0.0)
  penalty = torch.sum(violation, dim=-1)

  joint_level_count = joint_site_z.shape[1] // finger_count
  for joint_level in range(joint_level_count):
    level_start = joint_level * finger_count
    level_slice = slice(level_start, level_start + primary_finger_count)
    joint_violation = (
      joint_target_z[:, level_slice] - joint_site_z[:, level_slice]
    ).clamp_min(0.0)
    penalty += torch.sum(joint_violation, dim=-1)
  return penalty


def _single_hand_primary_finger_skeleton_points(
  env: ManagerBasedRlEnv,
  fingertip_cfg: SceneEntityCfg,
  joint_site_cfg: SceneEntityCfg,
  samples_per_segment: int,
) -> torch.Tensor:
  robot: Entity = env.scene[fingertip_cfg.name]
  tip_pos = robot.data.site_pos_w[:, fingertip_cfg.site_ids]
  joint_site_pos = robot.data.site_pos_w[:, joint_site_cfg.site_ids]

  finger_count = 5
  primary_finger_count = 3
  if tip_pos.shape[1] < primary_finger_count:
    raise ValueError("Fingertip site count must include thumb/index/middle.")
  if joint_site_pos.shape[1] % finger_count != 0:
    raise ValueError("Joint site count must be a multiple of 5 fingers.")
  if joint_site_pos.shape[1] // finger_count < 3:
    raise ValueError("Joint site cfg must include joint1/joint3/joint4 levels.")

  points: list[torch.Tensor] = []
  for finger_id in range(primary_finger_count):
    anchors = (
      joint_site_pos[:, finger_id],
      joint_site_pos[:, finger_count + finger_id],
      joint_site_pos[:, 2 * finger_count + finger_id],
      tip_pos[:, finger_id],
    )
    points.extend(anchors)
    for start, end in zip(anchors[:-1], anchors[1:]):
      for sample_id in range(samples_per_segment):
        alpha = float(sample_id + 1) / float(samples_per_segment + 1)
        points.append(start + alpha * (end - start))
  return torch.stack(points, dim=1)


def _world_points_to_object_local(
  env: ManagerBasedRlEnv,
  points_w: torch.Tensor,
  object_name: str,
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  batch_size, point_count, _ = points_w.shape
  rel = points_w - obj.data.root_link_pos_w[:, None, :]
  rel_flat = rel.reshape(batch_size * point_count, 3)
  object_quat_inv = quat_inv(obj.data.root_link_quat_w)
  quat_flat = object_quat_inv.repeat_interleave(point_count, dim=0)
  points_obj = quat_apply(quat_flat, rel_flat)
  return points_obj.reshape(batch_size, point_count, 3)


def _site_positions_in_object_local(
  env: ManagerBasedRlEnv,
  site_cfg: SceneEntityCfg,
  object_name: str,
) -> torch.Tensor:
  entity: Entity = env.scene[site_cfg.name]
  site_pos_w = entity.data.site_pos_w[:, site_cfg.site_ids]
  return _world_points_to_object_local(env, site_pos_w, object_name)


class contact_move_relative_pose_reward:
  """Keep hand key sites at the grasp-time relative pose to each object."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.activation_hold_steps = int(cfg.params["activation_hold_steps"])
    self.exp_scale = float(cfg.params["exp_scale"])
    self.include_left = bool(cfg.params.get("include_left", True))
    self.include_right = bool(cfg.params.get("include_right", True))
    self.left_gate_hold_counter = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self.right_gate_hold_counter = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self.left_unlocked = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self.right_unlocked = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self.left_has_reference = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self.right_has_reference = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self.left_reference: torch.Tensor | None = None
    self.right_reference: torch.Tensor | None = None
    episode_keys: list[str] = []
    if self.include_left:
      episode_keys.append("contact_move_relative_pose_left")
    if self.include_right:
      episode_keys.append("contact_move_relative_pose_right")
    self.episode_sums = _make_episode_sum_buffers(env, tuple(episode_keys))
    self.env = env

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.left_gate_hold_counter[env_ids] = 0
    self.right_gate_hold_counter[env_ids] = 0
    self.left_unlocked[env_ids] = False
    self.right_unlocked[env_ids] = False
    self.left_has_reference[env_ids] = False
    self.right_has_reference[env_ids] = False
    if self.left_reference is not None:
      self.left_reference[env_ids] = 0.0
    if self.right_reference is not None:
      self.right_reference[env_ids] = 0.0
    _flush_episode_sum_buffers(self.env, self.episode_sums, env_ids)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    left_fingertip_cfg: SceneEntityCfg,
    left_target_cfg: SceneEntityCfg,
    right_fingertip_cfg: SceneEntityCfg,
    right_target_cfg: SceneEntityCfg,
    left_sensor_name: str,
    right_sensor_name: str,
    activation_threshold: float,
    activation_force_threshold: float,
    activation_hold_steps: int,
    left_relative_site_cfg: SceneEntityCfg,
    right_relative_site_cfg: SceneEntityCfg,
    left_object_name: str,
    right_object_name: str,
    exp_scale: float,
    include_left: bool = True,
    include_right: bool = True,
  ) -> torch.Tensor:
    del activation_hold_steps, exp_scale
    include_left = self.include_left and bool(include_left)
    include_right = self.include_right and bool(include_right)

    if include_left:
      left_gate_mask = _single_hand_grasp_gate_mask(
        env,
        fingertip_cfg=left_fingertip_cfg,
        target_cfg=left_target_cfg,
        activation_threshold=activation_threshold,
        sensor_name=left_sensor_name,
        activation_force_threshold=activation_force_threshold,
      )
    else:
      left_gate_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if include_right:
      right_gate_mask = _single_hand_grasp_gate_mask(
        env,
        fingertip_cfg=right_fingertip_cfg,
        target_cfg=right_target_cfg,
        activation_threshold=activation_threshold,
        sensor_name=right_sensor_name,
        activation_force_threshold=activation_force_threshold,
      )
    else:
      right_gate_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if include_left:
      next_left_counter, next_left_unlocked = _update_gate_state(
        self.left_gate_hold_counter,
        self.left_unlocked,
        left_gate_mask,
        self.activation_hold_steps,
      )
      left_newly_unlocked = next_left_unlocked & ~self.left_unlocked
      self.left_gate_hold_counter = next_left_counter
      self.left_unlocked = next_left_unlocked
    else:
      left_newly_unlocked = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
      )

    if include_right:
      next_right_counter, next_right_unlocked = _update_gate_state(
        self.right_gate_hold_counter,
        self.right_unlocked,
        right_gate_mask,
        self.activation_hold_steps,
      )
      right_newly_unlocked = next_right_unlocked & ~self.right_unlocked
      self.right_gate_hold_counter = next_right_counter
      self.right_unlocked = next_right_unlocked
    else:
      right_newly_unlocked = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
      )

    left_relative: torch.Tensor | None = None
    right_relative: torch.Tensor | None = None
    if include_left:
      left_relative = _site_positions_in_object_local(
        env, left_relative_site_cfg, left_object_name
      )
      if self.left_reference is None:
        self.left_reference = torch.zeros_like(left_relative)
      if left_newly_unlocked.any():
        self.left_reference[left_newly_unlocked] = left_relative[left_newly_unlocked]
        self.left_has_reference[left_newly_unlocked] = True
    if include_right:
      right_relative = _site_positions_in_object_local(
        env, right_relative_site_cfg, right_object_name
      )
      if self.right_reference is None:
        self.right_reference = torch.zeros_like(right_relative)
      if right_newly_unlocked.any():
        self.right_reference[right_newly_unlocked] = right_relative[right_newly_unlocked]
        self.right_has_reference[right_newly_unlocked] = True

    if include_left and left_relative is not None and self.left_reference is not None:
      left_active = self.left_unlocked & self.left_has_reference
      left_error = torch.linalg.norm(left_relative - self.left_reference, dim=-1).mean(
        dim=-1
      )
      left_reward = torch.where(
        left_active,
        torch.exp(-self.exp_scale * left_error),
        torch.zeros_like(left_error),
      )
    else:
      left_reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if include_right and right_relative is not None and self.right_reference is not None:
      right_active = self.right_unlocked & self.right_has_reference
      right_error = torch.linalg.norm(
        right_relative - self.right_reference, dim=-1
      ).mean(dim=-1)
      right_reward = torch.where(
        right_active,
        torch.exp(-self.exp_scale * right_error),
        torch.zeros_like(right_error),
      )
    else:
      right_reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    step_scale = getattr(env, "step_dt", 1.0)
    if "contact_move_relative_pose_left" in self.episode_sums:
      self.episode_sums["contact_move_relative_pose_left"] += step_scale * left_reward
    if "contact_move_relative_pose_right" in self.episode_sums:
      self.episode_sums["contact_move_relative_pose_right"] += step_scale * right_reward
    active_sides = int(include_left) + int(include_right)
    if active_sides == 0:
      return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    return (left_reward + right_reward) / float(active_sides)


def _single_hand_object_sdf_penalty(
  env: ManagerBasedRlEnv,
  sdf_grid: _ObjectSdfGrid,
  object_name: str,
  fingertip_cfg: SceneEntityCfg,
  joint_site_cfg: SceneEntityCfg,
  clearance: float | torch.Tensor,
  samples_per_segment: int,
) -> torch.Tensor:
  points_w = _single_hand_primary_finger_skeleton_points(
    env,
    fingertip_cfg=fingertip_cfg,
    joint_site_cfg=joint_site_cfg,
    samples_per_segment=samples_per_segment,
  )
  points_obj = _world_points_to_object_local(env, points_w, object_name)
  sdf = sdf_grid.query(points_obj)
  if not torch.is_tensor(clearance):
    clearance = torch.tensor(float(clearance), dtype=sdf.dtype, device=sdf.device)
  if clearance.ndim == 1:
    clearance = clearance.unsqueeze(-1)
  return (clearance - sdf).clamp_min(0.0).mean(dim=-1)


def _single_hand_primary_fingertip_target_distance(
  env: ManagerBasedRlEnv,
  fingertip_cfg: SceneEntityCfg,
  target_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot: Entity = env.scene[fingertip_cfg.name]
  obj: Entity = env.scene[target_cfg.name]
  tip_pos = robot.data.site_pos_w[:, fingertip_cfg.site_ids]
  target_pos = obj.data.site_pos_w[:, target_cfg.site_ids]

  if tip_pos.shape[1] != target_pos.shape[1]:
    raise ValueError("Fingertip/target site count mismatch.")
  primary_finger_count = 3
  if tip_pos.shape[1] < primary_finger_count:
    raise ValueError("Fingertip site count must include thumb/index/middle.")

  primary_dist = torch.linalg.norm(
    tip_pos[:, :primary_finger_count] - target_pos[:, :primary_finger_count],
    dim=-1,
  )
  return primary_dist.mean(dim=-1)


def _early_step_penalty_multiplier(
  env: ManagerBasedRlEnv,
  step_window: int,
  max_multiplier: float,
) -> torch.Tensor:
  if step_window <= 0 or max_multiplier <= 1.0:
    return torch.ones(env.num_envs, dtype=torch.float32, device=env.device)
  step = env.episode_length_buf.to(dtype=torch.float32)
  remaining_ratio = ((float(step_window) - step) / float(step_window)).clamp(
    min=0.0, max=1.0
  )
  return 1.0 + (float(max_multiplier) - 1.0) * remaining_ratio


def _resolve_hand_action_ids(
  env: ManagerBasedRlEnv,
  action_term_name: str,
  hand_prefix: str,
) -> torch.Tensor:
  action_term = env.action_manager.get_term(action_term_name)
  target_names = getattr(action_term, "target_names", None)
  if target_names is None:
    raise AttributeError(
      f"Action term '{action_term_name}' does not expose target_names."
    )

  action_ids = [
    action_id
    for action_id, name in enumerate(target_names)
    if name.startswith(f"{hand_prefix}_")
  ]
  if not action_ids:
    raise ValueError(f"No action targets found for hand prefix '{hand_prefix}_'.")
  return torch.tensor(action_ids, device=env.device, dtype=torch.long)


def _single_hand_core_reward(
  env: ManagerBasedRlEnv,
  *,
  fingertip_cfg: SceneEntityCfg,
  target_cfg: SceneEntityCfg,
  wrist_cfg: SceneEntityCfg,
  wrist_target_cfg: SceneEntityCfg,
  fingertip_scales: tuple[float, ...],
  wrist_scale: float,
  velocity_scale: float,
  acceleration_scale: float,
  action_rate_scale: float,
  action_acceleration_scale: float,
  joint_asset_cfg: SceneEntityCfg,
  action_ids: torch.Tensor,
) -> torch.Tensor:
  alignment_reward = _single_hand_fingertip_wrist_target_alignment_reward(
    env,
    fingertip_cfg=fingertip_cfg,
    target_cfg=target_cfg,
    wrist_cfg=wrist_cfg,
    wrist_target_cfg=wrist_target_cfg,
    scales=fingertip_scales,
    wrist_scale=wrist_scale,
  )
  velocity_reward = joint_velocity_reward(
    env, scale=velocity_scale, asset_cfg=joint_asset_cfg
  )
  acceleration_reward = joint_acceleration_reward(
    env, scale=acceleration_scale, asset_cfg=joint_asset_cfg
  )
  action_reward = _action_rate_reward_subset(
    env, scale=action_rate_scale, action_ids=action_ids
  )
  action_acceleration_reward = _action_acceleration_reward_subset(
    env, scale=action_acceleration_scale, action_ids=action_ids
  )
  return (
    alignment_reward
    * velocity_reward
    * acceleration_reward
    * action_reward
    * action_acceleration_reward
  )


def _make_episode_sum_buffers(
  env: ManagerBasedRlEnv,
  keys: tuple[str, ...],
) -> dict[str, torch.Tensor]:
  return {
    key: torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    for key in keys
  }


def _make_episode_min_buffers(
  env: ManagerBasedRlEnv,
  keys: tuple[str, ...],
) -> dict[str, torch.Tensor]:
  return {
    key: torch.full(
      (env.num_envs,), float("inf"), dtype=torch.float32, device=env.device
    )
    for key in keys
  }


def _flush_episode_sum_buffers(
  env: ManagerBasedRlEnv,
  episode_sums: dict[str, torch.Tensor],
  env_ids: torch.Tensor | slice,
) -> None:
  for key, values in episode_sums.items():
    episodic_sum_avg = torch.mean(values[env_ids])
    env.extras["log"][f"Episode_Reward/{key}"] = (
      episodic_sum_avg / env.max_episode_length_s
    )
    values[env_ids] = 0.0


def _flush_episode_min_buffers(
  env: ManagerBasedRlEnv,
  episode_mins: dict[str, torch.Tensor],
  env_ids: torch.Tensor | slice,
) -> None:
  for key, values in episode_mins.items():
    episodic_min = torch.min(values[env_ids])
    episodic_min = torch.where(
      torch.isfinite(episodic_min),
      episodic_min,
      torch.zeros_like(episodic_min),
    )
    env.extras["log"][f"Episode_Reward/{key}"] = episodic_min
    values[env_ids] = float("inf")


class left_hand_internal_reward:
  """Left-hand internal reward: multiplicative core plus hand-local additives."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.env = env
    self.term_weight = float(cfg.weight)
    self.action_term_name = str(cfg.params.get("action_term_name", "joint_pos"))
    self.action_ids: torch.Tensor | None = None
    self.object_sdf_grid = _ObjectSdfGrid(str(cfg.params["object_sdf_file"]), env.device)
    self.object_sdf_samples_per_segment = int(
      cfg.params.get("object_sdf_samples_per_segment", 2)
    )
    self.episode_sums = _make_episode_sum_buffers(
      env,
      (
        "left_hand_internal_factor_alignment",
        "left_hand_internal_factor_joint_velocity",
        "left_hand_internal_factor_joint_acceleration",
        "left_hand_internal_factor_action_rate",
        "left_hand_internal_factor_action_acceleration",
        "left_hand_internal_factor_action_magnitude",
        "left_hand_internal_term_joint_site_alignment_primary",
        "left_hand_internal_term_joint_site_alignment_secondary",
        "left_hand_internal_term_site_height",
        "left_hand_internal_term_object_sdf",
        "left_hand_internal_term_joint_limit",
        "left_hand_internal_term_palm_ground",
      ),
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    _flush_episode_sum_buffers(self.env, self.episode_sums, env_ids)

  def _get_action_ids(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    if self.action_ids is None:
      self.action_ids = _resolve_hand_action_ids(
        env, action_term_name=self.action_term_name, hand_prefix="left"
      )
    return self.action_ids

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    fingertip_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    wrist_cfg: SceneEntityCfg,
    wrist_target_cfg: SceneEntityCfg,
    joint_site_cfg: SceneEntityCfg,
    joint_target_cfg: SceneEntityCfg,
    fingertip_scales: tuple[float, ...],
    wrist_scale: float,
    joint_site_scale: float,
    joint_site_primary_reward_weight: float,
    joint_site_secondary_reward_weight: float,
    site_height_penalty_weight: float,
    object_sdf_file: str,
    object_sdf_name: str,
    object_sdf_early_clearance: float,
    object_sdf_later_clearance: float,
    object_sdf_early_penalty_weight: float,
    object_sdf_later_penalty_weight: float,
    object_sdf_later_threshold: float,
    object_sdf_samples_per_segment: int,
    object_sdf_early_step_window: int,
    object_sdf_early_penalty_multiplier: float,
    core_reward_weight: float,
    velocity_scale: float,
    acceleration_scale: float,
    action_rate_scale: float,
    action_acceleration_scale: float,
    action_magnitude_scale: float,
    joint_asset_cfg: SceneEntityCfg,
    joint_limit_penalty_weight: float,
    palm_sensor_names: tuple[str, ...],
    palm_force_penalty_weight: float,
    action_term_name: str = "joint_pos",
  ) -> torch.Tensor:
    del action_term_name, object_sdf_file, object_sdf_samples_per_segment

    alignment_reward = _single_hand_fingertip_wrist_target_alignment_reward(
      env,
      fingertip_cfg=fingertip_cfg,
      target_cfg=target_cfg,
      wrist_cfg=wrist_cfg,
      wrist_target_cfg=wrist_target_cfg,
      scales=fingertip_scales,
      wrist_scale=wrist_scale,
    )
    (
      joint_site_alignment_primary_reward,
      joint_site_alignment_secondary_reward,
    ) = _single_hand_joint_site_target_alignment_reward(
      env,
      joint_site_cfg=joint_site_cfg,
      joint_target_cfg=joint_target_cfg,
      scale=joint_site_scale,
    )
    velocity_reward = joint_velocity_reward(
      env, scale=velocity_scale, asset_cfg=joint_asset_cfg
    )
    acceleration_reward = joint_acceleration_reward(
      env, scale=acceleration_scale, asset_cfg=joint_asset_cfg
    )
    action_reward = _action_rate_reward_subset(
      env, scale=action_rate_scale, action_ids=self._get_action_ids(env)
    )
    action_acceleration_reward = _action_acceleration_reward_subset(
      env,
      scale=action_acceleration_scale,
      action_ids=self._get_action_ids(env),
    )
    action_magnitude_reward = _action_magnitude_reward_subset(
      env,
      scale=action_magnitude_scale,
      action_ids=self._get_action_ids(env),
    )
    core_reward = (
      alignment_reward
      * velocity_reward
      * acceleration_reward
      * action_reward
      * action_acceleration_reward
      * action_magnitude_reward
    )
    core_term = core_reward_weight * core_reward
    joint_site_primary_term = (
      joint_site_primary_reward_weight
      * joint_site_alignment_primary_reward
    )
    joint_site_secondary_term = (
      joint_site_secondary_reward_weight
      * joint_site_alignment_secondary_reward
    )
    site_height_term = site_height_penalty_weight * (
      _single_hand_primary_site_height_penalty(
        env,
        fingertip_cfg=fingertip_cfg,
        target_cfg=target_cfg,
        joint_site_cfg=joint_site_cfg,
        joint_target_cfg=joint_target_cfg,
      )
    )
    object_sdf_target_distance = _single_hand_primary_fingertip_target_distance(
      env,
      fingertip_cfg=fingertip_cfg,
      target_cfg=target_cfg,
    )
    object_sdf_is_later = object_sdf_target_distance < object_sdf_later_threshold
    object_sdf_clearance = torch.where(
      object_sdf_is_later,
      torch.full_like(object_sdf_target_distance, object_sdf_later_clearance),
      torch.full_like(object_sdf_target_distance, object_sdf_early_clearance),
    )
    object_sdf_penalty = _single_hand_object_sdf_penalty(
      env,
      sdf_grid=self.object_sdf_grid,
      object_name=object_sdf_name,
      fingertip_cfg=fingertip_cfg,
      joint_site_cfg=joint_site_cfg,
      clearance=object_sdf_clearance,
      samples_per_segment=self.object_sdf_samples_per_segment,
    )
    object_sdf_weight = torch.where(
      object_sdf_is_later,
      torch.full_like(object_sdf_penalty, object_sdf_later_penalty_weight),
      torch.full_like(object_sdf_penalty, object_sdf_early_penalty_weight),
    )
    object_sdf_weight = object_sdf_weight * _early_step_penalty_multiplier(
      env,
      step_window=object_sdf_early_step_window,
      max_multiplier=object_sdf_early_penalty_multiplier,
    )
    object_sdf_term = object_sdf_weight * object_sdf_penalty
    joint_limit_term = joint_limit_penalty_weight * _joint_pos_limit_penalty(
      env, asset_cfg=joint_asset_cfg
    )
    palm_force_term = palm_force_penalty_weight * palm_ground_contact_force_penalty(
      env, sensor_names=palm_sensor_names
    )

    step_scale = self.term_weight * env.step_dt
    self.episode_sums["left_hand_internal_factor_alignment"] += (
      step_scale * alignment_reward
    )
    self.episode_sums["left_hand_internal_term_joint_site_alignment_primary"] += (
      step_scale * joint_site_primary_term
    )
    self.episode_sums["left_hand_internal_term_joint_site_alignment_secondary"] += (
      step_scale * joint_site_secondary_term
    )
    self.episode_sums["left_hand_internal_term_site_height"] += (
      step_scale * site_height_term
    )
    self.episode_sums["left_hand_internal_term_object_sdf"] += (
      step_scale * object_sdf_term
    )
    self.episode_sums["left_hand_internal_factor_joint_velocity"] += (
      step_scale * velocity_reward
    )
    self.episode_sums["left_hand_internal_factor_joint_acceleration"] += (
      step_scale * acceleration_reward
    )
    self.episode_sums["left_hand_internal_factor_action_rate"] += (
      step_scale * action_reward
    )
    self.episode_sums["left_hand_internal_factor_action_acceleration"] += (
      step_scale * action_acceleration_reward
    )
    self.episode_sums["left_hand_internal_factor_action_magnitude"] += (
      step_scale * action_magnitude_reward
    )
    self.episode_sums["left_hand_internal_term_joint_limit"] += (
      step_scale * joint_limit_term
    )
    self.episode_sums["left_hand_internal_term_palm_ground"] += (
      step_scale * palm_force_term
    )

    return (
      core_term
      + joint_site_primary_term
      + joint_site_secondary_term
      + site_height_term
      + object_sdf_term
      + joint_limit_term
      + palm_force_term
    )


class right_hand_internal_reward:
  """Right-hand internal reward: multiplicative core plus hand-local additives."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.env = env
    self.term_weight = float(cfg.weight)
    self.action_term_name = str(cfg.params.get("action_term_name", "joint_pos"))
    self.action_ids: torch.Tensor | None = None
    self.object_sdf_grid = _ObjectSdfGrid(str(cfg.params["object_sdf_file"]), env.device)
    self.object_sdf_samples_per_segment = int(
      cfg.params.get("object_sdf_samples_per_segment", 2)
    )
    self.episode_sums = _make_episode_sum_buffers(
      env,
      (
        "right_hand_internal_factor_alignment",
        "right_hand_internal_factor_joint_velocity",
        "right_hand_internal_factor_joint_acceleration",
        "right_hand_internal_factor_action_rate",
        "right_hand_internal_factor_action_acceleration",
        "right_hand_internal_factor_action_magnitude",
        "right_hand_internal_term_joint_site_alignment_primary",
        "right_hand_internal_term_joint_site_alignment_secondary",
        "right_hand_internal_term_site_height",
        "right_hand_internal_term_object_sdf",
        "right_hand_internal_term_joint_limit",
        "right_hand_internal_term_palm_ground",
      ),
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    _flush_episode_sum_buffers(self.env, self.episode_sums, env_ids)

  def _get_action_ids(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    if self.action_ids is None:
      self.action_ids = _resolve_hand_action_ids(
        env, action_term_name=self.action_term_name, hand_prefix="right"
      )
    return self.action_ids

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    fingertip_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    wrist_cfg: SceneEntityCfg,
    wrist_target_cfg: SceneEntityCfg,
    joint_site_cfg: SceneEntityCfg,
    joint_target_cfg: SceneEntityCfg,
    fingertip_scales: tuple[float, ...],
    wrist_scale: float,
    joint_site_scale: float,
    joint_site_primary_reward_weight: float,
    joint_site_secondary_reward_weight: float,
    site_height_penalty_weight: float,
    object_sdf_file: str,
    object_sdf_name: str,
    object_sdf_early_clearance: float,
    object_sdf_later_clearance: float,
    object_sdf_early_penalty_weight: float,
    object_sdf_later_penalty_weight: float,
    object_sdf_later_threshold: float,
    object_sdf_samples_per_segment: int,
    object_sdf_early_step_window: int,
    object_sdf_early_penalty_multiplier: float,
    core_reward_weight: float,
    velocity_scale: float,
    acceleration_scale: float,
    action_rate_scale: float,
    action_acceleration_scale: float,
    action_magnitude_scale: float,
    joint_asset_cfg: SceneEntityCfg,
    joint_limit_penalty_weight: float,
    palm_sensor_names: tuple[str, ...],
    palm_force_penalty_weight: float,
    action_term_name: str = "joint_pos",
  ) -> torch.Tensor:
    del action_term_name, object_sdf_file, object_sdf_samples_per_segment

    alignment_reward = _single_hand_fingertip_wrist_target_alignment_reward(
      env,
      fingertip_cfg=fingertip_cfg,
      target_cfg=target_cfg,
      wrist_cfg=wrist_cfg,
      wrist_target_cfg=wrist_target_cfg,
      scales=fingertip_scales,
      wrist_scale=wrist_scale,
    )
    (
      joint_site_alignment_primary_reward,
      joint_site_alignment_secondary_reward,
    ) = _single_hand_joint_site_target_alignment_reward(
      env,
      joint_site_cfg=joint_site_cfg,
      joint_target_cfg=joint_target_cfg,
      scale=joint_site_scale,
    )
    velocity_reward = joint_velocity_reward(
      env, scale=velocity_scale, asset_cfg=joint_asset_cfg
    )
    acceleration_reward = joint_acceleration_reward(
      env, scale=acceleration_scale, asset_cfg=joint_asset_cfg
    )
    action_reward = _action_rate_reward_subset(
      env, scale=action_rate_scale, action_ids=self._get_action_ids(env)
    )
    action_acceleration_reward = _action_acceleration_reward_subset(
      env,
      scale=action_acceleration_scale,
      action_ids=self._get_action_ids(env),
    )
    action_magnitude_reward = _action_magnitude_reward_subset(
      env,
      scale=action_magnitude_scale,
      action_ids=self._get_action_ids(env),
    )
    core_reward = (
      alignment_reward
      * velocity_reward
      * acceleration_reward
      * action_reward
      * action_acceleration_reward
      * action_magnitude_reward
    )
    core_term = core_reward_weight * core_reward
    joint_site_primary_term = (
      joint_site_primary_reward_weight
      * joint_site_alignment_primary_reward
    )
    joint_site_secondary_term = (
      joint_site_secondary_reward_weight
      * joint_site_alignment_secondary_reward
    )
    site_height_term = site_height_penalty_weight * (
      _single_hand_primary_site_height_penalty(
        env,
        fingertip_cfg=fingertip_cfg,
        target_cfg=target_cfg,
        joint_site_cfg=joint_site_cfg,
        joint_target_cfg=joint_target_cfg,
      )
    )
    object_sdf_target_distance = _single_hand_primary_fingertip_target_distance(
      env,
      fingertip_cfg=fingertip_cfg,
      target_cfg=target_cfg,
    )
    object_sdf_is_later = object_sdf_target_distance < object_sdf_later_threshold
    object_sdf_clearance = torch.where(
      object_sdf_is_later,
      torch.full_like(object_sdf_target_distance, object_sdf_later_clearance),
      torch.full_like(object_sdf_target_distance, object_sdf_early_clearance),
    )
    object_sdf_penalty = _single_hand_object_sdf_penalty(
      env,
      sdf_grid=self.object_sdf_grid,
      object_name=object_sdf_name,
      fingertip_cfg=fingertip_cfg,
      joint_site_cfg=joint_site_cfg,
      clearance=object_sdf_clearance,
      samples_per_segment=self.object_sdf_samples_per_segment,
    )
    object_sdf_weight = torch.where(
      object_sdf_is_later,
      torch.full_like(object_sdf_penalty, object_sdf_later_penalty_weight),
      torch.full_like(object_sdf_penalty, object_sdf_early_penalty_weight),
    )
    object_sdf_weight = object_sdf_weight * _early_step_penalty_multiplier(
      env,
      step_window=object_sdf_early_step_window,
      max_multiplier=object_sdf_early_penalty_multiplier,
    )
    object_sdf_term = object_sdf_weight * object_sdf_penalty
    joint_limit_term = joint_limit_penalty_weight * _joint_pos_limit_penalty(
      env, asset_cfg=joint_asset_cfg
    )
    palm_force_term = palm_force_penalty_weight * palm_ground_contact_force_penalty(
      env, sensor_names=palm_sensor_names
    )

    step_scale = self.term_weight * env.step_dt
    self.episode_sums["right_hand_internal_factor_alignment"] += (
      step_scale * alignment_reward
    )
    self.episode_sums["right_hand_internal_term_joint_site_alignment_primary"] += (
      step_scale * joint_site_primary_term
    )
    self.episode_sums["right_hand_internal_term_joint_site_alignment_secondary"] += (
      step_scale * joint_site_secondary_term
    )
    self.episode_sums["right_hand_internal_term_site_height"] += (
      step_scale * site_height_term
    )
    self.episode_sums["right_hand_internal_term_object_sdf"] += (
      step_scale * object_sdf_term
    )
    self.episode_sums["right_hand_internal_factor_joint_velocity"] += (
      step_scale * velocity_reward
    )
    self.episode_sums["right_hand_internal_factor_joint_acceleration"] += (
      step_scale * acceleration_reward
    )
    self.episode_sums["right_hand_internal_factor_action_rate"] += (
      step_scale * action_reward
    )
    self.episode_sums["right_hand_internal_factor_action_acceleration"] += (
      step_scale * action_acceleration_reward
    )
    self.episode_sums["right_hand_internal_factor_action_magnitude"] += (
      step_scale * action_magnitude_reward
    )
    self.episode_sums["right_hand_internal_term_joint_limit"] += (
      step_scale * joint_limit_term
    )
    self.episode_sums["right_hand_internal_term_palm_ground"] += (
      step_scale * palm_force_term
    )

    return (
      core_term
      + joint_site_primary_term
      + joint_site_secondary_term
      + site_height_term
      + object_sdf_term
      + joint_limit_term
      + palm_force_term
    )


def multiplicative_reward_residual(
  env: ManagerBasedRlEnv,
  left_fingertip_cfg: SceneEntityCfg,
  left_target_cfg: SceneEntityCfg,
  right_fingertip_cfg: SceneEntityCfg,
  right_target_cfg: SceneEntityCfg,
  left_wrist_cfg: SceneEntityCfg,
  left_wrist_target_cfg: SceneEntityCfg,
  right_wrist_cfg: SceneEntityCfg,
  right_wrist_target_cfg: SceneEntityCfg,
  left_sensor_name: str,
  right_sensor_name: str,
  left_target_graph: tuple[float, ...],
  right_target_graph: tuple[float, ...],
  distance_threshold: float,
  force_threshold: float,
  left_scales: tuple[float, ...],
  right_scales: tuple[float, ...],
  left_wrist_scale: float,
  right_wrist_scale: float,
  velocity_scale: float,
  acceleration_scale: float,
  action_rate_scale: float,
  smoothness_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Residual term so Episode_Reward terms can be shown without changing total objective."""
  fingertip_wrist_alignment_reward = fingertip_wrist_target_alignment_reward(
    env,
    left_fingertip_cfg=left_fingertip_cfg,
    left_target_cfg=left_target_cfg,
    right_fingertip_cfg=right_fingertip_cfg,
    right_target_cfg=right_target_cfg,
    left_wrist_cfg=left_wrist_cfg,
    left_wrist_target_cfg=left_wrist_target_cfg,
    right_wrist_cfg=right_wrist_cfg,
    right_wrist_target_cfg=right_wrist_target_cfg,
    left_scales=left_scales,
    right_scales=right_scales,
    left_wrist_scale=left_wrist_scale,
    right_wrist_scale=right_wrist_scale,
  )
  contact_graph_alignment_reward = fixed_contact_graph_alignment_reward(
    env,
    left_fingertip_cfg=left_fingertip_cfg,
    left_target_cfg=left_target_cfg,
    right_fingertip_cfg=right_fingertip_cfg,
    right_target_cfg=right_target_cfg,
    left_sensor_name=left_sensor_name,
    right_sensor_name=right_sensor_name,
    left_target_graph=left_target_graph,
    right_target_graph=right_target_graph,
    distance_threshold=distance_threshold,
    force_threshold=force_threshold,
  )
  joint_velocity_smoothness_reward = joint_velocity_reward(
    env, scale=velocity_scale, asset_cfg=smoothness_asset_cfg
  )
  joint_acceleration_smoothness_reward = joint_acceleration_reward(
    env, scale=acceleration_scale, asset_cfg=smoothness_asset_cfg
  )
  action_delta_smoothness_reward = action_rate_reward(env, scale=action_rate_scale)
  product_reward = (
    fingertip_wrist_alignment_reward
    * contact_graph_alignment_reward
    * joint_velocity_smoothness_reward
    * joint_acceleration_smoothness_reward
    * action_delta_smoothness_reward
  )
  additive_positive_sum = (
    fingertip_wrist_alignment_reward
    + contact_graph_alignment_reward
    + joint_velocity_smoothness_reward
    + joint_acceleration_smoothness_reward
    + action_delta_smoothness_reward
  )
  return product_reward - additive_positive_sum
