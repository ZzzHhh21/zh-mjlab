"""Collect stage-A rollout states for later reset-state filtering."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class CollectStageAConfig:
  checkpoint_file: str
  output_file: str = "logs/stage_a_states.pt"
  num_envs: int = 1024
  num_steps: int = 60
  device: str | None = None
  seed: int | None = None
  no_terminations: bool = False
  include_observations: bool = False


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


def _append_snapshot(
  storage: dict[str, list[torch.Tensor]],
  *,
  env,
  action: torch.Tensor | None,
  reward: torch.Tensor | None,
  done: torch.Tensor | None,
  obs,
) -> None:
  unwrapped = env.unwrapped
  robot = unwrapped.scene["robot"]
  left_obj = unwrapped.scene["left_obj"]
  right_obj = unwrapped.scene["right_obj"]
  num_envs = unwrapped.num_envs
  device = unwrapped.device

  storage["robot_joint_pos"].append(robot.data.joint_pos.detach().cpu())
  storage["robot_joint_vel"].append(robot.data.joint_vel.detach().cpu())
  storage["robot_root_state"].append(_root_state(robot).detach().cpu())
  storage["left_obj_root_state"].append(_root_state(left_obj).detach().cpu())
  storage["right_obj_root_state"].append(_root_state(right_obj).detach().cpu())
  storage["episode_length"].append(unwrapped.episode_length_buf.detach().cpu())
  storage["env_origins"].append(unwrapped.scene.env_origins.detach().cpu())

  if action is None:
    action = torch.zeros(num_envs, env.num_actions, device=device)
  if reward is None:
    reward = torch.zeros(num_envs, device=device)
  if done is None:
    done = torch.zeros(num_envs, device=device, dtype=torch.bool)
  storage["action"].append(action.detach().cpu())
  storage["reward"].append(reward.detach().cpu())
  storage["done"].append(done.to(dtype=torch.bool).detach().cpu())

  if obs is not None:
    for key, value in obs.items():
      if torch.is_tensor(value):
        storage.setdefault(f"obs/{key}", []).append(value.detach().cpu())


def _stack_storage(storage: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
  return {key: torch.stack(values, dim=0) for key, values in storage.items()}


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


def run_collect(task_id: str, cfg: CollectStageAConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = int(cfg.num_envs)
  if cfg.seed is not None:
    env_cfg.seed = int(cfg.seed)
    agent_cfg.seed = int(cfg.seed)
  if cfg.no_terminations:
    env_cfg.terminations = {}

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  checkpoint_path = Path(cfg.checkpoint_file).expanduser()
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)

  obs = env.get_observations()
  storage: dict[str, list[torch.Tensor]] = {}
  if not cfg.include_observations:
    obs_for_storage = None
  else:
    obs_for_storage = obs
  _append_snapshot(
    storage,
    env=env,
    action=None,
    reward=None,
    done=None,
    obs=obs_for_storage,
  )

  with torch.inference_mode():
    for step in range(int(cfg.num_steps)):
      action = policy(obs).detach()
      obs, reward, done, _ = env.step(action)
      obs_for_storage = obs if cfg.include_observations else None
      _append_snapshot(
        storage,
        env=env,
        action=action,
        reward=reward,
        done=done,
        obs=obs_for_storage,
      )
      if (step + 1) % 10 == 0 or step + 1 == int(cfg.num_steps):
        done_rate = done.to(dtype=torch.float32).mean().item()
        print(
          f"[INFO] collected step {step + 1}/{cfg.num_steps}, "
          f"done_rate={done_rate:.4f}",
          flush=True,
        )

  output = {
    "metadata": {
      "task_id": task_id,
      "checkpoint_file": str(checkpoint_path),
      "num_envs": int(cfg.num_envs),
      "num_steps": int(cfg.num_steps),
      "device": device,
      "seed": cfg.seed,
      "no_terminations": bool(cfg.no_terminations),
      "include_observations": bool(cfg.include_observations),
      "action_target_names": _action_target_names(env),
    },
    "data": _stack_storage(storage),
  }

  output_path = Path(cfg.output_file).expanduser()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(output, output_path)
  print(f"[INFO] saved stage-A states to: {output_path}")

  env.close()


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(
    CollectStageAConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_collect(task_id=chosen_task, cfg=cfg)


if __name__ == "__main__":
  main()
