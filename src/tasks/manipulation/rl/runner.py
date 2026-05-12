import os
import statistics
import time

import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.utils import check_nan
from rsl_rl.utils.logger import Logger

_HIDDEN_LOG_KEYS = {
  "Episode_Reward/contact_move_object_regularization",
  "Episode_Reward/contact_move_relative_pose",
  "Episode_Reward/left_hand_internal",
  "Episode_Reward/multiplicative_reward_residual",
  "Episode_Reward/right_hand_internal",
  "Episode_Termination/right_object_stage_success_count",
}

_ORDERED_LOG_GROUPS = (
  (
    "Episode_Reward/key_fingertip_target_distance",
    "Episode_Reward/contact_move_right_speed",
    "Episode_Reward/right_object_reset_rotation_penalty",
  ),
  (
    "Episode_Reward/right_object_stage_segment_progress",
    "Episode_Reward/right_object_stage_waypoint_exp",
    "Episode_Reward/right_object_stage_height_reward",
    "Episode_Reward/right_object_stage_max_waypoint_index",
    "Episode_Reward/right_object_stage",
  ),
)


def _filter_and_reorder_log(log: dict) -> dict:
  filtered = {key: value for key, value in log.items() if key not in _HIDDEN_LOG_KEYS}
  grouped_keys = {key for group in _ORDERED_LOG_GROUPS for key in group}
  emitted_groups: set[int] = set()
  reordered = {}
  for key, value in filtered.items():
    group_idx = next(
      (
        idx
        for idx, group in enumerate(_ORDERED_LOG_GROUPS)
        if key in group
      ),
      None,
    )
    if group_idx is None:
      reordered[key] = value
      continue
    if group_idx in emitted_groups:
      continue
    for group_key in _ORDERED_LOG_GROUPS[group_idx]:
      if group_key in filtered:
        reordered[group_key] = filtered[group_key]
    emitted_groups.add(group_idx)
  for key, value in filtered.items():
    if key not in grouped_keys and key not in reordered:
      reordered[key] = value
  return reordered


class _ManipulationLogger(Logger):
  """Task-local logger that can hide low-signal debug reward terms."""

  def process_env_step(
    self,
    rewards,
    dones,
    extras,
    intrinsic_rewards=None,
  ) -> None:
    if "log" in extras:
      extras = dict(extras)
      extras["log"] = _filter_and_reorder_log(dict(extras["log"]))

    if "episode" in extras:
      extras = dict(extras)
      extras["episode"] = _filter_and_reorder_log(dict(extras["episode"]))
    super().process_env_step(rewards, dones, extras, intrinsic_rewards)


class ManipulationOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env,
    train_cfg,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device)
    self.logger = _ManipulationLogger(
      log_dir=self.logger.log_dir,
      cfg=self.logger.cfg,
      env_cfg=self.logger.env_cfg,
      num_envs=self.logger.num_envs,
      is_distributed=self.is_distributed,
      gpu_world_size=self.gpu_world_size,
      gpu_global_rank=self.gpu_global_rank,
      device=self.device,
    )
    self.best_mean_reward = float("-inf")
    self.best_mean_reward_iteration = -1

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    """Run learning and keep an overwritten best-reward checkpoint."""
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()

    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(obs, rewards, dones, extras)
          intrinsic_rewards = (
            self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
          )
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

        stop = time.time()
        collect_time = stop - start
        start = stop
        self.alg.compute_returns(obs)

      loss_dict = self.alg.update()

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
      )

      self._save_best_checkpoint_if_needed(it)

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

    if self.logger.writer is not None:
      self.save(
        os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
      )  # type: ignore[arg-type]
      self.logger.stop_logging_writer()

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ):
    infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
    if infos:
      self.best_mean_reward = float(infos.get("best_mean_reward", self.best_mean_reward))
      self.best_mean_reward_iteration = int(
        infos.get("best_mean_reward_iteration", self.best_mean_reward_iteration)
      )
    return infos

  def _save_best_checkpoint_if_needed(self, it: int) -> None:
    if self.logger.writer is None or self.logger.log_dir is None:
      return
    if len(self.logger.rewbuffer) == 0:
      return

    mean_reward = float(statistics.mean(self.logger.rewbuffer))
    if mean_reward <= self.best_mean_reward:
      return

    self.best_mean_reward = mean_reward
    self.best_mean_reward_iteration = int(it)
    best_path = os.path.join(self.logger.log_dir, "model_best.pt")
    self._save_checkpoint_only(
      best_path,
      infos={
        "best_mean_reward": self.best_mean_reward,
        "best_mean_reward_iteration": self.best_mean_reward_iteration,
      },
    )
    print(
      "[INFO] Saved new best checkpoint: "
      f"{best_path} (mean_reward={mean_reward:.4f}, iteration={it})"
    )

  def _save_checkpoint_only(self, path: str, infos=None) -> None:
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {
      **(infos or {}),
      "best_mean_reward": self.best_mean_reward,
      "best_mean_reward_iteration": self.best_mean_reward_iteration,
      "env_state": env_state,
    }
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)

  def save(self, path: str, infos=None):
    infos = {
      **(infos or {}),
      "best_mean_reward": self.best_mean_reward,
      "best_mean_reward_iteration": self.best_mean_reward_iteration,
    }
    super().save(path, infos)
    policy_dir = os.path.dirname(path)
    filename = "policy.onnx"
    onnx_path = os.path.join(policy_dir, filename)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(
          str(onnx_path),
          base_path=str(policy_dir),
        )
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
