from src.tasks.manipulation.rl import ManipulationOnPolicyRunner
import mjlab.tasks.registry as _task_registry
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  unitree_g1_wuji_contact_env_cfg,
  unitree_g1_wuji_lift_cube_env_cfg,
  unitree_g1_wuji_move_env_cfg,
)
from .rl_cfg import unitree_g1_wuji_manip_ppo_runner_cfg

_REMOVED_TASK_IDS = (
  "Mjlab-Lift-Cube-Yam",
  "Mjlab-Lift-Cube-Yam-Rgb",
  "Mjlab-Lift-Cube-Yam-Depth",
  "Mjlab-Multi-Cube-Seg-Yam",
)

for _task_id in _REMOVED_TASK_IDS:
  _task_registry._REGISTRY.pop(_task_id, None)  # noqa: SLF001

register_mjlab_task(
  task_id="Unitree-G1-Wuji",
  env_cfg=unitree_g1_wuji_lift_cube_env_cfg(),
  play_env_cfg=unitree_g1_wuji_lift_cube_env_cfg(play=True),
  rl_cfg=unitree_g1_wuji_manip_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Wuji-Contact",
  env_cfg=unitree_g1_wuji_contact_env_cfg(),
  play_env_cfg=unitree_g1_wuji_contact_env_cfg(play=True),
  rl_cfg=unitree_g1_wuji_manip_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Wuji-Move",
  env_cfg=unitree_g1_wuji_move_env_cfg(),
  play_env_cfg=unitree_g1_wuji_move_env_cfg(play=True),
  rl_cfg=unitree_g1_wuji_manip_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
