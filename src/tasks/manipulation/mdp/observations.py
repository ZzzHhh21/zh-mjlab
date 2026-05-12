from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_inv
from .rewards import (
  _advance_stage_index,
  _right_hand_grasp_gate_mask,
  _update_gate_state,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _as_tuple(value: str | tuple[str, ...]) -> tuple[str, ...]:
  return (value,) if isinstance(value, str) else value


def ee_to_object_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance vector(s) from end effector site(s) to object in base frame.

  Single site: shape ``(B, 3)``. Multiple sites (e.g. bimanual): vectors are
  concatenated along the last axis, shape ``(B, 3 * N)``.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids]
  obj_pos_w = obj.data.root_link_pos_w
  base_quat_w = robot.data.root_link_quat_w
  n = ee_pos_w.shape[1]
  if n == 0:
    # Fallback for robots/tasks without configured EE sites.
    # Keep 6-D shape for bimanual checkpoints by duplicating the same vector.
    distance_vec_w = obj_pos_w - robot.data.root_link_pos_w
    distance_vec_b = quat_apply(quat_inv(base_quat_w), distance_vec_w)
    return torch.cat((distance_vec_b, distance_vec_b), dim=-1)
  parts = []
  for i in range(n):
    distance_vec_w = obj_pos_w - ee_pos_w[:, i, :]
    distance_vec_b = quat_apply(quat_inv(base_quat_w), distance_vec_w)
    parts.append(distance_vec_b)
  if n == 1:
    return parts[0]
  return torch.cat(parts, dim=-1)


def object_positions(
  env: ManagerBasedRlEnv,
  object_names: str | tuple[str, ...],
) -> torch.Tensor:
  """Object root positions in each environment's local world frame."""
  positions = []
  env_origins = env.scene.env_origins
  for name in _as_tuple(object_names):
    obj: Entity = env.scene[name]
    positions.append(obj.data.root_link_pos_w - env_origins)
  return torch.cat(positions, dim=-1)


def object_quaternions(
  env: ManagerBasedRlEnv,
  object_names: str | tuple[str, ...],
) -> torch.Tensor:
  """Object root orientations as quaternions."""
  quaternions = []
  for name in _as_tuple(object_names):
    obj: Entity = env.scene[name]
    quaternions.append(obj.data.root_link_quat_w)
  return torch.cat(quaternions, dim=-1)


def fingertip_contact_forces(
  env: ManagerBasedRlEnv,
  sensor_name: str | tuple[str, ...],
  include_magnitude: bool = True,
  log_scale: bool = True,
) -> torch.Tensor:
  """Fingertip contact force vectors, optionally with force magnitude."""
  parts = []
  for name in _as_tuple(sensor_name):
    sensor: ContactSensor = env.scene[name]
    force = sensor.data.force
    assert force is not None
    if log_scale:
      force = torch.sign(force) * torch.log1p(torch.abs(force))
    if include_magnitude:
      magnitude = torch.linalg.norm(sensor.data.force, dim=-1, keepdim=True)
      if log_scale:
        magnitude = torch.log1p(magnitude)
      force = torch.cat((force, magnitude), dim=-1)
    parts.append(force.flatten(start_dim=1))
  return torch.cat(parts, dim=-1)


def object_to_fingertip_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  fingertip_cfg: SceneEntityCfg,
  include_distance: bool = True,
) -> torch.Tensor:
  """Vectors from object root to fingertip sites, optionally with distances."""
  robot: Entity = env.scene[fingertip_cfg.name]
  obj: Entity = env.scene[object_name]
  fingertip_pos = robot.data.site_pos_w[:, fingertip_cfg.site_ids]
  relative_pos = fingertip_pos - obj.data.root_link_pos_w[:, None, :]
  if include_distance:
    distance = torch.linalg.norm(relative_pos, dim=-1, keepdim=True)
    relative_pos = torch.cat((relative_pos, distance), dim=-1)
  return relative_pos.flatten(start_dim=1)


def site_to_site_relative(
  env: ManagerBasedRlEnv,
  source_cfg: SceneEntityCfg,
  target_cfg: SceneEntityCfg,
  include_distance: bool = True,
) -> torch.Tensor:
  """Vectors from source sites to matching target sites, optionally with distances."""
  source: Entity = env.scene[source_cfg.name]
  target: Entity = env.scene[target_cfg.name]
  source_pos = source.data.site_pos_w[:, source_cfg.site_ids]
  target_pos = target.data.site_pos_w[:, target_cfg.site_ids]
  if source_pos.shape[1] != target_pos.shape[1]:
    raise ValueError("Source/target site count mismatch.")
  relative_pos = target_pos - source_pos
  if include_distance:
    distance = torch.linalg.norm(relative_pos, dim=-1, keepdim=True)
    relative_pos = torch.cat((relative_pos, distance), dim=-1)
  return relative_pos.flatten(start_dim=1)


def right_object_stage_target_relative(
  env: ManagerBasedRlEnv,
  moving_site_cfg: SceneEntityCfg,
  target_site_cfg: SceneEntityCfg,
  include_distance: bool = True,
) -> torch.Tensor:
  """Vectors from the right-object moving site to all ordered stage targets."""
  moving_obj: Entity = env.scene[moving_site_cfg.name]
  target_obj: Entity = env.scene[target_site_cfg.name]
  moving_pos = moving_obj.data.site_pos_w[:, moving_site_cfg.site_ids]
  target_pos = target_obj.data.site_pos_w[:, target_site_cfg.site_ids]
  if moving_pos.shape[1] != 1:
    raise ValueError("Moving-site cfg must contain exactly one site.")
  relative_pos = target_pos - moving_pos
  if include_distance:
    distance = torch.linalg.norm(relative_pos, dim=-1, keepdim=True)
    relative_pos = torch.cat((relative_pos, distance), dim=-1)
  return relative_pos.flatten(start_dim=1)


class right_object_stage_state:
  """Observable task-progress state for the staged right-object objective."""

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self.activation_hold_steps = int(cfg.params["activation_hold_steps"])
    self.stage_thresholds = torch.tensor(
      cfg.params["stage_thresholds"], device=env.device, dtype=torch.float32
    )
    self.gate_hold_counter = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self.unlocked = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self.stage_index = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.gate_hold_counter[env_ids] = 0
    self.unlocked[env_ids] = False
    self.stage_index[env_ids] = 0

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
  ) -> torch.Tensor:
    del activation_hold_steps, stage_thresholds, stage_exp_scales

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

    moving_obj: Entity = env.scene[moving_site_cfg.name]
    target_obj: Entity = env.scene[stage_target_cfg.name]
    moving_pos = moving_obj.data.site_pos_w[:, moving_site_cfg.site_ids]
    target_pos = target_obj.data.site_pos_w[:, stage_target_cfg.site_ids]
    stage_distances = torch.linalg.norm(target_pos - moving_pos, dim=-1)
    self.stage_index = _advance_stage_index(
      self.unlocked,
      self.stage_index,
      stage_distances,
      self.stage_thresholds,
    )

    hold_progress = self.gate_hold_counter.to(torch.float32) / float(
      self.activation_hold_steps
    )
    hold_progress = hold_progress.clamp(max=1.0)
    return torch.stack(
      (
        gate_mask.float(),
        hold_progress,
        self.unlocked.float(),
        self.stage_index.to(torch.float32),
      ),
      dim=-1,
    )


def ee_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """EE linear velocity in EE frame."""
  robot: Entity = env.scene[asset_cfg.name]
  ee_vel_w = robot.data.site_vel_w[:, asset_cfg.site_ids].squeeze(1)  # (B, 6)
  ee_vel_linear_w = ee_vel_w[:, :3]
  ee_quat_w = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
  ee_vel_linear_ee = quat_apply(quat_inv(ee_quat_w), ee_vel_linear_w)
  return ee_vel_linear_ee
