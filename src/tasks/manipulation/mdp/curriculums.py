from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


# Stage schemas.


class _RewardCurriculumStageOptional(TypedDict, total=False):
  weight: float
  params: dict[str, Any]


class RewardCurriculumStage(_RewardCurriculumStageOptional):
  step: int


class _TerminationCurriculumStageOptional(TypedDict, total=False):
  params: dict[str, Any]
  time_out: bool


class TerminationCurriculumStage(_TerminationCurriculumStageOptional):
  step: int


# Shared engine.  Stage dicts are passed directly from the public TypedDict
# schemas.  Any key that isn't "step" or "params" is treated as a top-level
# field on the target term config (e.g. "weight" on RewardTermCfg).

_RESERVED_KEYS = {"step", "params"}


def _validate_stages(
  term_cfg: Any,
  term_name: str,
  stages: Sequence[Any],
) -> None:
  """Validate stage ordering, field existence, and param keys."""
  for i in range(1, len(stages)):
    if stages[i]["step"] < stages[i - 1]["step"]:
      raise ValueError(
        f"Curriculum stages must be in nondecreasing step order,"
        f" but stage {i} has step"
        f" {stages[i]['step']} < {stages[i - 1]['step']}."
      )
  for stage in stages:
    for key in stage:
      if key not in _RESERVED_KEYS and not hasattr(term_cfg, key):
        raise AttributeError(
          f"Field '{key}' does not exist on the resolved term config for '{term_name}'."
        )
  for stage in stages:
    unknown = stage.get("params", {}).keys() - term_cfg.params.keys()
    if unknown:
      raise KeyError(
        f"Stage at step {stage['step']} sets unknown param(s)"
        f" {unknown} on term '{term_name}'. Check for typos."
      )


def _apply_stages(
  term_cfg: Any,
  step_counter: int,
  stages: Sequence[Any],
) -> dict[str, torch.Tensor]:
  """Apply staged updates and return a logging snapshot."""
  for stage in stages:
    if step_counter >= stage["step"]:
      for key, value in stage.items():
        if key not in _RESERVED_KEYS:
          setattr(term_cfg, key, value)
      if "params" in stage:
        term_cfg.params.update(stage["params"])
  # Only log values that stages actually reference.
  logged_fields: set[str] = set()
  logged_params: set[str] = set()
  for stage in stages:
    for key in stage:
      if key not in _RESERVED_KEYS:
        logged_fields.add(key)
    for key in stage.get("params", {}):
      logged_params.add(key)
  result: dict[str, torch.Tensor] = {}
  for key in logged_fields:
    value = getattr(term_cfg, key)
    if isinstance(value, (int, float, bool)):
      result[key] = torch.tensor(value)
    elif isinstance(value, torch.Tensor):
      result[key] = value
  for key in logged_params:
    v = term_cfg.params[key]
    if isinstance(v, (int, float, bool)):
      result[key] = torch.tensor(v)
    elif isinstance(v, torch.Tensor):
      result[key] = v
  return result


# Public wrappers.


def _resolve_target_site_ids(
  env: ManagerBasedRlEnv,
  target_specs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
  targets: list[dict[str, Any]] = []
  for spec in target_specs:
    target_entity_name = str(spec["target_entity_name"])
    target_site_name = str(spec["target_site_name"])
    target_entity = env.scene[target_entity_name]
    site_ids, _ = target_entity.find_sites(
      (target_site_name,),
      preserve_order=True,
    )
    if len(site_ids) != 1:
      raise ValueError(
        "Target-y curriculum expects each target spec to contain exactly one site."
      )
    targets.append(
      {
        "site_id": int(target_entity.indexing.site_ids[site_ids[0]].item()),
        "start_y": float(spec["start_y"]),
        "end_y": float(spec["end_y"]),
        "log_key": str(spec.get("log_key", f"{target_site_name}_y")),
      }
    )
  return targets


def _target_y(start_y: float, end_y: float, difficulty: float) -> float:
  return start_y + (end_y - start_y) * difficulty


def _apply_target_y_values(
  env: ManagerBasedRlEnv,
  targets: Sequence[dict[str, Any]],
  difficulty: float,
) -> dict[str, float]:
  target_ys: dict[str, float] = {}
  for target in targets:
    target_y = _target_y(target["start_y"], target["end_y"], difficulty)
    site_id = int(target["site_id"])
    env.sim.model.site_pos[:, site_id, 1] = target_y
    env.sim.mj_model.site_pos[site_id, 1] = target_y
    target_ys[str(target["log_key"])] = target_y
  return target_ys


def set_right_object_target_difficulty(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  targets: Sequence[dict[str, Any]],
  difficulty: float,
) -> None:
  """Set fixed target2/target3 local-y difficulty, used by play startup."""
  del env_ids
  resolved_targets = _resolve_target_site_ids(env, targets)
  env.sim.expand_model_fields(("site_pos",))
  _apply_target_y_values(env, resolved_targets, float(difficulty))


class reward_curriculum:
  """Update a reward term's weight and/or params based on training steps.

  Each stage specifies a ``step`` threshold and optionally a ``weight``
  and/or ``params`` dict.  When ``env.common_step_counter`` reaches a
  stage's ``step``, the corresponding values are applied.  Later stages
  take precedence when multiple thresholds are reached.

  Example::

    CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_hinge",
        "stages": [
          {"step": 0, "weight": -0.01},
          {"step": 12000, "weight": -0.1},
          {"step": 24000, "weight": -1.0, "params": {"max_vel": 1.0}},
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    reward_name: str = cfg.params["reward_name"]
    stages: list[RewardCurriculumStage] = cfg.params["stages"]
    self._term_cfg = env.reward_manager.get_term_cfg(reward_name)
    self._stages = stages
    _validate_stages(self._term_cfg, reward_name, self._stages)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    stages: list[RewardCurriculumStage],
  ) -> dict[str, torch.Tensor]:
    del env_ids, reward_name, stages
    return _apply_stages(self._term_cfg, env.common_step_counter, self._stages)


class termination_curriculum:
  """Update a termination term's params based on training steps.

  Each stage specifies a ``step`` threshold and a ``params`` dict.  When
  ``env.common_step_counter`` reaches a stage's ``step``, the params are
  applied.  Later stages take precedence.

  Example::

    CurriculumTermCfg(
      func=mdp.termination_curriculum,
      params={
        "termination_name": "energy",
        "stages": [
          {"step": 12000, "params": {"threshold": 1000.0}},
          {"step": 24000, "params": {"threshold": 700.0}},
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    termination_name: str = cfg.params["termination_name"]
    stages: list[TerminationCurriculumStage] = cfg.params["stages"]
    self._term_cfg = env.termination_manager.get_term_cfg(termination_name)
    self._stages = stages
    _validate_stages(self._term_cfg, termination_name, self._stages)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    termination_name: str,
    stages: list[TerminationCurriculumStage],
  ) -> dict[str, torch.Tensor]:
    del env_ids, termination_name, stages
    return _apply_stages(self._term_cfg, env.common_step_counter, self._stages)


class right_object_target3_curriculum:
  """Increase target2/target3 difficulty when recent success rate stays high."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self.success_threshold = float(cfg.params["success_threshold"])
    self.required_consecutive = int(cfg.params.get("required_consecutive", 1))
    self.difficulty_step = float(cfg.params["difficulty_step"])
    self.max_difficulty = float(cfg.params.get("max_difficulty", 1.0))
    self.difficulty = 0.0
    self.consecutive_success_batches = 0
    target_specs = cfg.params.get("targets")
    if target_specs is None:
      target_specs = (
        {
          "target_entity_name": cfg.params["target_entity_name"],
          "target_site_name": cfg.params["target_site_name"],
          "start_y": cfg.params["start_y"],
          "end_y": cfg.params["end_y"],
          "log_key": "target3_y",
        },
      )
    self.targets = _resolve_target_site_ids(env, target_specs)
    env.sim.expand_model_fields(("site_pos",))
    self._apply_target_y(env)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    targets: tuple[dict[str, Any], ...] | None = None,
    success_threshold: float = 0.0,
    required_consecutive: int = 1,
    difficulty_step: float = 0.1,
    max_difficulty: float = 1.0,
    **_: Any,
  ) -> dict[str, torch.Tensor]:
    del targets, success_threshold, required_consecutive, difficulty_step, max_difficulty
    success_rate = self._success_rate(env, env_ids)
    if success_rate > self.success_threshold:
      self.consecutive_success_batches += 1
    else:
      self.consecutive_success_batches = 0

    if self.consecutive_success_batches >= self.required_consecutive:
      old_difficulty = self.difficulty
      self.difficulty = min(
        self.max_difficulty,
        self.difficulty + self.difficulty_step,
      )
      if self.difficulty > old_difficulty:
        self.consecutive_success_batches = 0

    target_ys = self._apply_target_y(env)
    logs = {
      "difficulty_level": torch.tensor(self.difficulty, device=env.device),
      "success_rate": torch.tensor(success_rate, device=env.device),
    }
    for key, value in target_ys.items():
      logs[key] = torch.tensor(value, device=env.device)
    return logs

  def _success_rate(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> float:
    term = env.termination_manager.get_term("right_object_stage_success_count")
    if env_ids is None or isinstance(env_ids, slice):
      values = term[env_ids]
    else:
      values = term[env_ids.to(device=env.device, dtype=torch.long)]
    if values.numel() == 0:
      return 0.0
    return float(torch.count_nonzero(values).item()) / float(values.numel())

  def _apply_target_y(self, env: ManagerBasedRlEnv) -> dict[str, float]:
    return _apply_target_y_values(env, self.targets, self.difficulty)
