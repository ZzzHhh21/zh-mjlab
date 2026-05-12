"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from src.tasks.manipulation.rl.residual_policy import (
  FrozenBasePolicy,
  ResidualPolicyConfig,
  ResidualPolicyVecEnvWrapper,
)

_RESIDUAL_MASKED_PPO_CLASS = "src.tasks.manipulation.rl.residual_ppo.ResidualMaskedPPO"


def _scale_or_default(value: float | None, fallback: float) -> float:
  return float(fallback if value is None else value)


def _log_robot_default_joint_positions(env: ManagerBasedRlEnv) -> None:
  """Print resolved default joint positions after regex expansion."""
  try:
    robot = env.scene["robot"]
    joint_ids, joint_names = robot.find_joints(".*")
    default_joint_pos = robot.data.default_joint_pos[0, joint_ids].detach().cpu()
    print("[INFO] Robot default joint positions (rad):")
    for name, value in zip(joint_names, default_joint_pos.tolist(), strict=True):
      print(f"[INFO]   {name}: {value:.6f}")
  except Exception as exc:  # pragma: no cover
    print(f"[WARN] Failed to log robot default joint positions: {exc}")


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  motion_file: str | None = None
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
  """Use recorded robot/object states as reset states during training."""
  stage_a_init_start_frame: int | None = None
  """Optional first recorded frame to sample from in stage_a_init_state_file."""
  stage_a_init_end_frame: int | None = None
  """Optional exclusive recorded frame end to sample from in stage_a_init_state_file."""
  residual_entropy_coef: float = 0.001
  residual_std_min: float = 0.05
  residual_std_max: float = 1.0
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def _load_frozen_policy(
  env,
  agent_cfg: dict,
  device: str,
  checkpoint_file: str,
  runner_cls,
  policy_label: str,
):
  checkpoint_path = Path(checkpoint_file).expanduser()
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"{policy_label} checkpoint not found: {checkpoint_path}")

  frozen_runner_cls = runner_cls or MjlabOnPolicyRunner
  # Frozen policies are inference-only references. In multi-GPU runs the real
  # training runner must be the only object that initializes torch.distributed;
  # constructing runner instances here with WORLD_SIZE>1 would initialize the
  # default process group before the main runner and then fail on the next one.
  original_world_size = os.environ.get("WORLD_SIZE")
  os.environ["WORLD_SIZE"] = "1"
  try:
    frozen_runner = frozen_runner_cls(env, deepcopy(agent_cfg), device=device)
  finally:
    if original_world_size is None:
      os.environ.pop("WORLD_SIZE", None)
    else:
      os.environ["WORLD_SIZE"] = original_world_size
  frozen_runner.load(
    str(checkpoint_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  print(f"[INFO] Frozen {policy_label} policy loaded from: {checkpoint_path}")
  return FrozenBasePolicy(frozen_runner.get_inference_policy(device=device))


def _sync_residual_train_start_with_env_cfg(
  env_cfg: ManagerBasedRlEnvCfg,
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


def _residual_train_start_step(cfg: TrainConfig) -> int:
  if cfg.residual_contact_checkpoint_file is None:
    return int(cfg.residual_base_steps)
  return int(cfg.residual_base_steps) + int(cfg.residual_contact_steps)


class StageAStateInitEnvWrapper:
  """Reset training envs from recorded robot joint states and object root states."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
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

    self._robot_joint_pos = self._robot_joint_pos.to(self.device)
    self._robot_joint_vel = self._robot_joint_vel.to(self.device)
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
      "[INFO] Stage-A init state bank enabled: "
      f"file={self.state_file}, states={self._num_states}, "
      f"joint_dim={self._robot_joint_pos.shape[-1]}, "
      f"object_root_states={self._has_object_root_states}, "
      f"frame_range=[{self._frame_start}, {self._frame_end})"
    )

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self.env.unwrapped

  @property
  def device(self) -> torch.device:
    return torch.device(self.unwrapped.device)

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


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

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
      raise ValueError("--stage-a-init-state-file requires residual training mode.")
    if cfg.residual_contact_checkpoint_file is None:
      raise ValueError(
        "--stage-a-init-state-file is intended for stage-C training and requires "
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
  if residual_mode:
    _sync_residual_train_start_with_env_cfg(
      cfg.env,
      _residual_train_start_step(cfg),
    )

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    if not cfg.motion_file:
      raise ValueError("For tracking tasks, --motion-file must be set ...")
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Using motion file: {motion_cmd.motion_file}")

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )
  if rank == 0:
    _log_robot_default_joint_positions(env)
  if cfg.stage_a_init_state_file is not None:
    env = StageAStateInitEnvWrapper(
      env,
      state_file=cfg.stage_a_init_state_file,
      start_frame=cfg.stage_a_init_start_frame,
      end_frame=cfg.stage_a_init_end_frame,
    )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
    # Load checkpoint from local filesystem.
    resume_path = get_checkpoint_path(
      log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
    )
    if cfg.residual_base_checkpoint_file is not None:
      base_checkpoint_path = Path(
        cfg.residual_base_checkpoint_file
      ).expanduser().resolve()
      if resume_path.expanduser().resolve() == base_checkpoint_path:
        raise ValueError(
          "Residual training cannot resume the residual actor from the same "
          "checkpoint used as the frozen base policy. Remove --agent.resume "
          "for a fresh residual run, or resume from a previous residual move run."
        )
    if cfg.residual_contact_checkpoint_file is not None:
      contact_checkpoint_path = Path(
        cfg.residual_contact_checkpoint_file
      ).expanduser().resolve()
      if resume_path.expanduser().resolve() == contact_checkpoint_path:
        raise ValueError(
          "Residual move training cannot resume the trainable actor from the "
          "same checkpoint used as the frozen contact policy. Resume from a "
          "previous move-residual run, or start a fresh move-residual run."
        )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  if residual_mode:
    if cfg.agent.resume:
      print(
        "[WARN] Residual mode with --agent.resume resumes the residual actor, "
        "not the frozen base actor. Do not point --agent.load-run to the base "
        "approach run unless you intentionally want to initialize the residual "
        "actor from that checkpoint."
      )
    base_policy = _load_frozen_policy(
      env=env,
      agent_cfg=agent_cfg,
      device=device,
      checkpoint_file=cfg.residual_base_checkpoint_file,
      runner_cls=runner_cls,
      policy_label="base",
    )
    contact_policy = None
    if cfg.residual_contact_checkpoint_file is not None:
      contact_policy = _load_frozen_policy(
        env=env,
        agent_cfg=agent_cfg,
        device=device,
        checkpoint_file=cfg.residual_contact_checkpoint_file,
        runner_cls=runner_cls,
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
    agent_cfg["algorithm"]["class_name"] = _RESIDUAL_MASKED_PPO_CLASS
    agent_cfg["algorithm"]["entropy_coef"] = float(cfg.residual_entropy_coef)
    agent_cfg["algorithm"]["residual_std_min"] = float(cfg.residual_std_min)
    agent_cfg["algorithm"]["residual_std_max"] = float(cfg.residual_std_max)
    print(
      "[INFO] Residual policy mode enabled: "
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
      f"train_start_step={_residual_train_start_step(cfg)}, "
      "base_contact_phase_training=masked, "
      f"entropy_coef={cfg.residual_entropy_coef}, "
      f"std=[{cfg.residual_std_min}, {cfg.residual_std_max}]"
    )

  runner_kwargs = {}
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))
    if residual_mode and hasattr(runner.alg, "_clamp_actor_std"):
      runner.alg._clamp_actor_std()

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations,
    init_at_random_ep_len=not residual_mode,
  )

  env.close()


def _assert_cuda_available_when_requested(
  gpu_ids: list[int] | Literal["all"] | None,
) -> None:
  """Fail early when GPU training was requested but CUDA is unavailable."""
  if gpu_ids is None:
    return

  import torch

  if torch.cuda.is_available():
    return

  raise RuntimeError(
    "GPU training was requested via --gpu-ids, but PyTorch cannot use CUDA. "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, "
    f"torch.version.cuda={torch.version.cuda!r}. "
    "Fix the NVIDIA driver / PyTorch CUDA mismatch, or pass --gpu-ids None "
    "to run on CPU intentionally."
  )


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.residual_base_checkpoint_file is not None:
    residual_stage_prefix = "C" if "right_object_stage" in args.env.rewards else "B"
    log_dir_name = f"{residual_stage_prefix}CC{log_dir_name}"
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  _assert_cuda_available_when_requested(args.gpu_ids)
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


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

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
