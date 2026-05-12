from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from src import SRC_PATH
from src.assets.robots.unitree_g1_wuji.g1_wuji_manip_constants import (
  G1_WUJI_MANIP_ACTION_SCALE,
  get_g1_wuji_manip_robot_cfg,
)
from src.tasks.manipulation import mdp as manipulation_mdp

FINGER_SITE_SUFFIXES = ("t", "i", "m", "r", "l")  # 五个手指的后缀顺序：thumb / index / middle / ring / little
TARGET_HAND_JOINT_NAMES = ("joint1", "joint3", "joint4")  # 物体侧目标关节点：近端 / 中端 / 远端
TARGET_HAND_JOINT_SITE_SIZE = 0.004

# 左右手目标点位置
LEFT_OBJ_LOCAL_TARGET_CONTACT_SITES = (
  ("left_target_contact_site_t", (-0.015627, 0.008012, -0.036261), (1.0, 0.0, 0.0, 1.0)), # y z x
  ("left_target_contact_site_i", (-0.000838, -0.002080, 0.040824), (0.0, 0.4, 1.0, 1.0)),
  ("left_target_contact_site_m", (0.010074, -0.022423, 0.045246), (0.0, 1.0, 0.0, 1.0)),
  ("left_target_contact_site_r", (0.025808, -0.027212, 0.041458), (1.0, 0.8, 0.0, 1.0)),
  ("left_target_contact_site_l", (0.041053, -0.033080, 0.043498), (1.0, 0.0, 1.0, 1.0)),
)
RIGHT_OBJ_LOCAL_TARGET_CONTACT_SITES = (
  ("right_target_contact_site_t", (0.012433, 0.030673, -0.014585), (1.0, 0.0, 0.0, 1.0)),
  ("right_target_contact_site_i", (-0.008284, 0.0350005, 0.014475), (0.0, 0.4, 1.0, 1.0)),
  ("right_target_contact_site_m", (-0.023668, 0.024306, -0.009500), (0.0, 1.0, 0.0, 1.0)),
  ("right_target_contact_site_r", (-0.050467, 0.030648, -0.003715), (1.0, 0.8, 0.0, 1.0)),
  ("right_target_contact_site_l", (-0.068554, 0.028508, -0.023691), (1.0, 0.0, 1.0, 1.0)),
)

LEFT_OBJ_LOCAL_TARGET_JOINT_SITES = (
  ("left_target_joint1_site_t", (0.075049, 0.020717, -0.060559), (1.0, 0.0, 0.0, 1.0)), #y z x
  ("left_target_joint1_site_i", (0.047624, 0.033900, 0.001152), (0.0, 0.4, 1.0, 1.0)),
  ("left_target_joint1_site_m", (0.065524, 0.022843, 0.000154), (0.0, 1.0, 0.0, 1.0)),
  ("left_target_joint1_site_r", (0.084046, 0.012130, -0.005858), (1.0, 0.8, 0.0, 1.0)),
  ("left_target_joint1_site_l", (0.102006, 0.000789, -0.005545), (1.0, 0.0, 1.0, 1.0)),

  ("left_target_joint3_site_t", (0.025997, 0.020168, -0.056268), (1.0, 0.0, 0.0, 1.0)),
  ("left_target_joint3_site_i", (0.015252, 0.030082, 0.033548), (0.0, 0.4, 1.0, 1.0)),
  ("left_target_joint3_site_m", (0.045524, 0.012843, 0.045154), (0.0, 1.0, 0.0, 1.0)),
  ("left_target_joint3_site_r", (0.053923, 0.001608, 0.037740), (1.0, 0.8, 0.0, 1.0)),
  ("left_target_joint3_site_l", (0.077977, -0.010987, 0.030993), (1.0, 0.0, 1.0, 1.0)),

  ("left_target_joint4_site_t", (0.004097, 0.020160, -0.048698), (1.0, 0.0, 0.0, 1.0)),
  ("left_target_joint4_site_i", (-0.000933, 0.013580, 0.048267), (0.0, 0.4, 1.0, 1.0)),
  ("left_target_joint4_site_m", (0.025170, -0.004647, 0.055110), (0.0, 1.0, 0.0, 1.0)),
  ("left_target_joint4_site_r", (0.032836, -0.013094, 0.052213), (1.0, 0.8, 0.0, 1.0)),
  ("left_target_joint4_site_l", (0.048798, -0.025924, 0.061130), (1.0, 0.0, 1.0, 1.0)),
)
RIGHT_OBJ_LOCAL_TARGET_JOINT_SITES = (
  ("right_target_joint1_site_t", (-0.013081, 0.064131, -0.102120), (1.0, 0.0, 0.0, 1.0)),
  ("right_target_joint1_site_i", (-0.016088, 0.095761, -0.040684), (0.0, 0.4, 1.0, 1.0)),
  ("right_target_joint1_site_m", (-0.037306, 0.090692, -0.048015), (0.0, 1.0, 0.0, 1.0)),
  ("right_target_joint1_site_r", (-0.062282, 0.076370, -0.044736), (1.0, 0.8, 0.0, 1.0)),
  ("right_target_joint1_site_l", (-0.070986, 0.062008, -0.054816), (1.0, 0.0, 1.0, 1.0)),

  ("right_target_joint3_site_t", (0.014183, 0.055573, -0.066296), (1.0, 0.0, 0.0, 1.0)),
  ("right_target_joint3_site_i", (-0.017194, 0.080972, -0.000035), (0.0, 0.4, 1.0, 1.0)),
  ("right_target_joint3_site_m", (-0.036531, 0.068316, -0.008390), (0.0, 1.0, 0.0, 1.0)),
  ("right_target_joint3_site_r", (-0.064808, 0.069828, -0.019373), (1.0, 0.8, 0.0, 1.0)),
  ("right_target_joint3_site_l", (-0.079220, 0.055873, -0.040475), (1.0, 0.0, 1.0, 1.0)),

  ("right_target_joint4_site_t", (0.016662, 0.044212, -0.035923), (1.0, 0.0, 0.0, 1.0)),
  ("right_target_joint4_site_i", (-0.013340, 0.055757, 0.019521), (0.0, 0.4, 1.0, 1.0)),
  ("right_target_joint4_site_m", (-0.031223, 0.040793, 0.001104), (0.0, 1.0, 0.0, 1.0)),
  ("right_target_joint4_site_r", (-0.060951, 0.053175, -0.005082), (1.0, 0.8, 0.0, 1.0)),
  ("right_target_joint4_site_l", (-0.080478, 0.043932, -0.027360), (1.0, 0.0, 1.0, 1.0)),
)

OBJECT_HAND_COLLISIONS_ENABLED = True  # 物体-手部碰撞开关：False 时物体不与手部碰撞，但仍可与地面/其他物体碰撞；True 时保留默认手-物碰撞
_OBJECT_COLLISION_BIT = 4

LEFT_OBJ_LOCAL_WRIST_TARGET_SITE = ("left_wrist_target_site", (0.106971, 0.010083, -0.067604), (1.0, 1.0, 1.0, 1.0),)
RIGHT_OBJ_LOCAL_WRIST_TARGET_SITE = ("right_wrist_target_site",(-0.020827, 0.060605, -0.118093),(0.7, 1.0, 1.0, 1.0))

# 物体移动阶段目标点
RIGHT_OBJ_LOCAL_POSITION_TARGET1 = (
  "right_obj_position_target1",
  (-0.05, 0.12, 0.0),
  (0.0, 0.0, 0.0, 1.0),
)
RIGHT_OBJ_LOCAL_POSITION_TARGET2 = (
  "right_obj_position_target2",
  (0.0, 0.09, 0.0),
  (0.0, 0.0, 0.0, 1.0),
)
RIGHT_OBJ_LOCAL_POSITION_TARGET3 = (
  "right_obj_position_target3",
  (0.0, 0.06, 0.0),
  (0.0, 0.0, 0.0, 1.0),
)
RIGHT_OBJ_LOCAL_POSITION_BEGIN = (
  "right_obj_position_begin",
  (0.0, 0.03, 0.0),
  (0.0, 0.0, 0.0, 1.0),
)

# episode 长度
TRAIN_EPISODE_LENGTH_S = 20.0                          # 训练 episode 时长：控制频率 50 Hz 下对应 1000 step
PLAY_EPISODE_LENGTH_S = 20.0                           # play episode 时长：保留较长时间方便观察

# 奖励判定阈值 / 门控条件
RIGHT_OBJECT_STAGE_ACTIVATION_THRESHOLD = 0.02          # 物体移动阶段奖励激活前提的距离阈值：右手拇指/食指/中指需进入各自目标 site 的 2 cm 范围内
RIGHT_OBJECT_STAGE_ACTIVATION_HOLD_STEPS = 10           # 物体移动阶段奖励激活前提的连续保持步数：右手关键指尖需连续满足“距离 + 接触力”条件 10 个 step 才解锁阶段奖励
RIGHT_OBJECT_STAGE_THRESHOLDS = (0.02, 0.01, 0.01)      # 物体移动三个阶段目标点（target1/2/3）的到点判定距离阈值，单位 m
RIGHT_OBJECT_STAGE_NO_PROGRESS_TERMINATION_STEPS = 150  # 阶段 C 解锁后若连续该步数 waypoint 不增长则终止，抑制“卡在高分段磨时间”
RIGHT_OBJECT_TARGET2_CURRICULUM_START_Y = 0.09          # target2 课程难度 0 时的 left_obj 局部 y 坐标
RIGHT_OBJECT_TARGET2_CURRICULUM_END_Y = 0.07            # target2 课程难度 1 时的 left_obj 局部 y 坐标
RIGHT_OBJECT_TARGET3_CURRICULUM_START_Y = 0.06          # target3 课程难度 0 时的 left_obj 局部 y 坐标
RIGHT_OBJECT_TARGET3_CURRICULUM_END_Y = 0.04            # target3 课程难度 1 时的 left_obj 局部 y 坐标
RIGHT_OBJECT_TARGET_CURRICULUM_SUCCESS_THRESHOLD = 0.7  # reset 批成功率连续高于该阈值时提高难度
RIGHT_OBJECT_TARGET_CURRICULUM_REQUIRED_CONSECUTIVE = 10 # 连续多少个 reset 批成功率达标后难度 + step
RIGHT_OBJECT_TARGET_CURRICULUM_DIFFICULTY_STEP = 0.1    # 每次达标后难度增量；难度范围 [0, 1]，只升不降
RIGHT_OBJECT_TARGET_PLAY_DIFFICULTY = 1.0               # play 固定目标点难度：0 为 target2/3 初始 y，1 为最高难度 y
CONTACT_GRAPH_DISTANCE_THRESHOLD = 0.02                 # 指尖-物体接触距离阈值：指尖 site 与物体对应目标接触 site 小于 2 cm 才满足距离条件
FINGERTIP_OBJECT_CONTACT_FORCE_THRESHOLD = 0.0          # 指尖-物体接触力阈值：用于接触图奖励与右手阶段奖励激活门控；接触力模长大于该值才满足“存在接触力”；当前为 0，表示只要有非零接触力就算接触
KEY_FINGERTIP_TARGET_DISTANCE_FAILURE_THRESHOLD = 0.02  # residual 阶段关键指尖保持阈值：超过目标 site 2 cm 后开始强惩罚
KEY_FINGERTIP_TARGET_DISTANCE_TERMINATION_THRESHOLD = 0.03 # residual 阶段关键指尖失败阈值：任一关键指尖超过目标 site 3 cm 视为失败
KEY_FINGERTIP_TARGET_DISTANCE_TERMINATION_ACTIVE_AFTER_STEPS = 50 # 关键指尖距离失败终止的默认启用步数；residual 训练/播放时会被 base_steps + contact_steps 覆盖

# 右物体移动路径插值
RIGHT_OBJECT_STAGE_INTERP_COUNTS = (3, 0, 0)            # begin->target1 插 3 个，target1->target2 不插值，target2->target3 不插值(2, 0, 1)

# 奖励指数 scale
FINGERTIP_TARGET_DISTANCE_SCALES = (5.0, 4.0, 3.0, 2.0, 1.0) # 五个指尖到目标接触点的 exp scale，按 thumb/index/middle/ring/little 顺序设置
WRIST_TARGET_DISTANCE_SCALE = 1.0                      # 手腕到目标点对齐奖励的指数 scale
HAND_JOINT_TARGET_DISTANCE_SCALE = 1.0                 # 手部 joint1/joint3/joint4 site 到对应目标 site 的指数 scale
JOINT_VELOCITY_SMOOTHNESS_EXP_SCALE = 1e-3             # 关节速度平滑奖励的指数 scale：速度范数越大，奖励衰减越快
JOINT_ACCELERATION_SMOOTHNESS_EXP_SCALE = 1e-3         # 关节加速度平滑奖励的指数 scale：加速度范数越大，奖励衰减越快
ACTION_RATE_SMOOTHNESS_EXP_SCALE = 1e-3                #1e-3 动作变化率平滑奖励的指数 scale：相邻 step 动作差越大，奖励衰减越快
ACTION_ACCELERATION_SMOOTHNESS_EXP_SCALE = 1e-3        # 动作二阶平滑奖励的指数 scale：a_t - 2 a_{t-1} + a_{t-2} 越大，奖励衰减越快
ACTION_MAGNITUDE_SMOOTHNESS_EXP_SCALE = 1e-3           # 动作幅值平滑奖励的指数 scale：sum(a_t^2) 越大，奖励衰减越快
CONTACT_GRAPH_EXP_SCALE = 1.0                          # 接触图 exp scale：reward = exp(-scale * sum(|current_contact - target_contact|))；error=1/2/3 时约为 0.37/0.14/0.05
KEY_FINGERTIP_FORCE_EXP_SCALE = 0.5                    # 关键指尖接触力奖励 scale：reward = exp(-scale / (sum_key_force + eps))；阶段 B 降低 scale 使接触力奖励更密集
CONTACT_MOVE_RELATIVE_POSE_EXP_SCALE = 20.0            # 相对位姿保持奖励 scale：reward = exp(-scale * mean_relative_error)
RIGHT_OBJECT_STAGE_EXP_SCALES = (5.0, 10.0, 10.0)        # 物体移动三个阶段奖励的指数 scale：r_i = exp(-scale_i * distance_i)，当前 0.01 m 时约为 exp(-0.05)
RIGHT_OBJECT_STAGE_SEGMENT_EXP_SCALE = 1.0             # 物体移动 segment 投影进度的横向偏离 scale；小于 waypoint scale，使沿段前进奖励更密集
RIGHT_OBJECT_STAGE_HEIGHT_EXP_SCALE = 5.0             # 物体移动高度跟踪 scale：reward = exp(-scale * |z - waypoint_z|)，抑制贴地只走投影方向
RIGHT_OBJECT_STAGE_HEIGHT_PROGRESS_MIX = 0.5           # 高度项中“朝当前 waypoint 高度前进”的占比；只到当前目标高度为止，超过后不继续增加
RIGHT_OBJECT_STAGE_HEIGHT_TRACK_MIX = 0.5              # 高度项中“贴近当前 waypoint 高度”的占比；高于或低于当前目标高度都会衰减

# 奖励 / 惩罚权重
HAND_INTERNAL_CORE_REWARD_WEIGHT = 2.0           # 左右手内部主乘积项权重：core = 对齐 * 速度平滑 * 加速度平滑 * 动作一阶平滑 * 动作二阶平滑
HAND_JOINT_PRIMARY_FINGER_REWARD_WEIGHT = 1.0          # 手部 joint 对齐奖励中拇指/食指/中指组的加法权重
HAND_JOINT_SECONDARY_FINGER_REWARD_WEIGHT = 0.2        # 手部 joint 对齐奖励中无名指/小指组的加法权重
HAND_SITE_HEIGHT_PENALTY_WEIGHT = -0.0                # 拇指/食指/中指的指尖与 joint134 site 低于对应目标 site 时的线性高度惩罚权重
HAND_OBJECT_SDF_EARLY_PENALTY_WEIGHT = -0.0          # 前期 STL SDF 虚拟穿透强惩罚权重，用于抑制快速穿透
HAND_OBJECT_SDF_LATER_PENALTY_WEIGHT = -0.0           # 之后 STL SDF 虚拟穿透弱惩罚权重，避免干扰最终贴近目标
CONTACT_GRAPH_REWARD_WEIGHT = 2.0                      # 接触图奖励权重：residual 阶段先强调建立多指稳定接触
KEY_FINGERTIP_FORCE_REWARD_WEIGHT = 2.0                # 关键指尖接触力奖励权重：阶段 B 优先让 thumb/index/middle 建立有效接触
KEY_FINGERTIP_TARGET_DISTANCE_PENALTY_WEIGHT = -5.0  # residual 阶段关键指尖远离目标点的强惩罚权重
CONTACT_MOVE_RELATIVE_POSE_REWARD_WEIGHT = 1.0         # 稳定接触 gate 解锁后，保持关键手指 site 与物体相对位置的奖励权重
CONTACT_MOVE_OBJECT_REGULARIZATION_REWARD_WEIGHT = 1.0 # 阶段 B/C 仍约束物体稳定，但降低早期接触尝试的压制
CONTACT_MOVE_RIGHT_SPEED_REWARD_WEIGHT = 1.0           # 阶段 C `contact_move_right_speed` 总体权重：在 right speed 子项上线性缩放 右物体速度惩罚
RIGHT_OBJECT_STAGE_SEGMENT_REWARD_WEIGHT = 1.5         # 右物体阶段 segment 投影进度权重；配合 1.5/1.5/3.0 使 dense 上限为 6
RIGHT_OBJECT_STAGE_WAYPOINT_REWARD_WEIGHT = 3.0        # 右物体阶段 waypoint 距离权重；配合 1.5/1.5/3.0 使 dense 上限为 6
RIGHT_OBJECT_STAGE_HEIGHT_REWARD_WEIGHT = 1.5          # 右物体阶段世界 z 高度跟踪权重；配合 1.5/1.5/3.0 使 dense 上限为 6
RIGHT_OBJECT_STAGE_PASSED_WAYPOINT_REWARD_WEIGHT = 1.0 # 每通过一个 waypoint 的阶段基线增量，保证下一段不低于上一段
RIGHT_OBJECT_STAGE_SUCCESS_REWARD_MULTIPLIER = RIGHT_OBJECT_STAGE_NO_PROGRESS_TERMINATION_STEPS     # 成功奖励 = multiplier * (W_seg + W_wp + W_h)，默认 12 > 中间步最大约 6
RIGHT_OBJECT_STAGE_REWARD_WEIGHT = 5.0                 # 阶段 C `right_object_stage` 总体权重：对整项阶段奖励线性缩放
RIGHT_OBJECT_RESET_ROTATION_PENALTY_THRESHOLD = math.radians(45.0) # 右物体相对 reset 姿态旋转容差，单位 rad；
RIGHT_OBJECT_RESET_ROTATION_PENALTY_WEIGHT = -10.0      # 右物体相对 reset 姿态旋转线性惩罚权重：允许右物体移动时有小幅旋转
JOINT_POSITION_LIMITS_REWARD_WEIGHT = -10.0            # 左右手内部各自对本侧关节子集计算的限位惩罚权重；作为加法负项，不参与乘法
PALM_GROUND_CONTACT_FORCE_PENALTY_WEIGHT = -0.01       # 左右手内部各自对本侧掌心-地面接触力计算的附加项权重

# STL SDF 虚拟穿透参数
HAND_OBJECT_SDF_LATER_THRESHOLD = 0.01                 # SDF 前期/之后切换阈值：primary 三指指尖平均目标距离小于该值后使用“之后”配置
HAND_OBJECT_SDF_EARLY_CLEARANCE = 0.02                 # 前期 STL SDF 安全距离，单位 m；更早触发惩罚以抑制起步穿透
HAND_OBJECT_SDF_LATER_CLEARANCE = 0.01                 # 之后 STL SDF 安全距离，单位 m；避免最终贴近目标时排斥过强
HAND_OBJECT_SDF_SAMPLES_PER_SEGMENT = 5                # primary 三指相邻关节点之间的插值采样点数
HAND_OBJECT_SDF_EARLY_STEP_WINDOW = 30                 # 起步阶段 SDF 惩罚放大的 step 窗口；用于重点抑制 episode 前期穿透
HAND_OBJECT_SDF_EARLY_PENALTY_MULTIPLIER = 5.0         # 起步第 0 step 的 SDF 惩罚倍率，随后在窗口内线性衰减到 1

# residual 接触/移动物体正则容差 / 限速
CONTACT_MOVE_OBJECT_LIFT_TOLERANCE = 0.01              # 物体允许的竖直抬起容差，超过后按线性高度惩罚
CONTACT_MOVE_OBJECT_LINEAR_VELOCITY_TOLERANCE = 0.02   # 阶段 B 允许的物体轻微线速度，超过后才惩罚
CONTACT_MOVE_OBJECT_ANGULAR_VELOCITY_TOLERANCE = 0.2   # 阶段 B 允许的物体轻微角速度，超过后才惩罚
LEFT_OBJECT_XY_DISPLACEMENT_TOLERANCE = 0.005          # 左物体允许的水平轻微扰动，超过后才惩罚
RIGHT_OBJECT_XY_DISPLACEMENT_TOLERANCE = 0.01          # 右物体允许的水平轻微扰动，超过后才惩罚
RIGHT_OBJECT_MOVE_LINEAR_VELOCITY_LIMIT = 0.10         # 阶段 C 右物体允许的移动线速度上限，超过后才惩罚；不鼓励静止
RIGHT_OBJECT_MOVE_ANGULAR_VELOCITY_LIMIT = 1.0         # 阶段 C 右物体允许的角速度上限，超过后才惩罚；用于防止打飞/高速旋转

# residual 接触/移动物体正则惩罚权重
LEFT_OBJECT_LINEAR_VELOCITY_PENALTY_WEIGHT = -2.0
LEFT_OBJECT_ANGULAR_VELOCITY_PENALTY_WEIGHT = -0.1
LEFT_OBJECT_XY_DISPLACEMENT_PENALTY_WEIGHT = -10.0
LEFT_OBJECT_LIFT_PENALTY_WEIGHT = -50.0
RIGHT_OBJECT_PRE_GRASP_LINEAR_VELOCITY_PENALTY_WEIGHT = -3.0
RIGHT_OBJECT_PRE_GRASP_ANGULAR_VELOCITY_PENALTY_WEIGHT = -0.2
RIGHT_OBJECT_XY_DISPLACEMENT_PENALTY_WEIGHT = -10.0
RIGHT_OBJECT_MOVE_LINEAR_VELOCITY_PENALTY_WEIGHT = -0.2
RIGHT_OBJECT_MOVE_ANGULAR_VELOCITY_PENALTY_WEIGHT = -0.05
RIGHT_OBJECT_LIFT_PENALTY_WEIGHT = -50.0

# 接触图目标配置
LEFT_TARGET_CONTACT_GRAPH = (1.0, 1.0, 0.0, 0.0, 0.0)   # 左手关键接触指尖掩码：thumb/index 参与接触奖励，middle/ring/little 在该项里忽略
RIGHT_TARGET_CONTACT_GRAPH = (1.0, 1.0, 1.0, 0.0, 0.0)  # 右手关键接触指尖掩码：thumb/index/middle 参与接触奖励，ring/little 在该项里忽略

# 域随机化
OBS_JOINT_POS_NOISE_RANGE = (-0.02, 0.02)            # 关节位置
OBS_JOINT_VEL_NOISE_RANGE = (-1.5, 1.5)
OBS_POSITION_NOISE_RANGE = (-0.005, 0.005)
OBS_QUATERNION_NOISE_RANGE = (-0.002, 0.002)
OBS_CONTACT_FORCE_NOISE_RANGE = (-0.02, 0.02)
INITIAL_JOINT_POSITION_RANDOM_RANGE = (-0.05, 0.05)
INITIAL_JOINT_VELOCITY_RANDOM_RANGE = (-0.1, 0.1)
PD_GAIN_RANDOM_SCALE_RANGE = (0.8, 1.2)

# 地面支撑面与物体随机范围
OBJECT_SUPPORT_Z = 0.81
OBJECT_SURFACE_CLEARANCE = 0.004
OBJECT_INITIAL_HEIGHT_OFFSET = -0.00  # 物体初始高度整体下调 1 cm；不改变支撑面高度
SPAWN_RECT_CENTER_X = 0.42  # 黄线中心
SPAWN_RECT_CENTER_Y = 0.0   # 黄线中心
SPAWN_RECT_SIZE_X = 0.3     # 黄线长度
SPAWN_RECT_SIZE_Y = 0.3     # 黄线宽度
SPAWN_RECT_LINE_THICKNESS = 0.004   # 黄线厚度
SPAWN_RECT_LINE_HALF_Z = 0.001   # 黄线高度
SPAWN_RECT_RGBA = (0.95, 0.9, 0.05, 1.0)   # 黄线颜色
SPAWN_RECT_VISUAL_ENV_LIMIT = 1   # 黄线可视化环境限制

SPAWN_REGION_X_RANGE = (0.42, 0.42)
LEFT_OBJ_SPAWN_Y_RANGE = (0.10, 0.10)
RIGHT_OBJ_SPAWN_Y_RANGE = (-0.10, -0.10)

# SPAWN_REGION_X_RANGE = (0.42, 0.42)
# LEFT_OBJ_SPAWN_Y_RANGE = (0.0, 0.0)
# RIGHT_OBJ_SPAWN_Y_RANGE = (-0.0, -0.0)

# SPAWN_REGION_X_RANGE = (
#   SPAWN_RECT_CENTER_X - SPAWN_RECT_SIZE_X / 2 + 0.05,
#   SPAWN_RECT_CENTER_X + SPAWN_RECT_SIZE_X / 2 - 0.05,
# ) # (0.42, 0.42)   -->  (0.32, 0.52)
# LEFT_OBJ_SPAWN_Y_RANGE = (
#   0.05,
#   SPAWN_RECT_CENTER_Y + SPAWN_RECT_SIZE_Y / 2 - 0.03,
# ) # (0.10, 0.10)   -->  (0.05, 0.12)  
# RIGHT_OBJ_SPAWN_Y_RANGE = (
#   SPAWN_RECT_CENTER_Y - SPAWN_RECT_SIZE_Y / 2 + 0.03,
#   -0.05,
# ) # (-0.10, -0.10) -->  (-0.12, -0.05) 

# 物体 reset 时的初始高度：zmin 已按 OBJ_INITIAL_QUAT 的初始姿态旋转后计算。
LEFT_COLLISION_ZMIN = -0.04939973
RIGHT_COLLISION_ZMIN = 0.01590837
LEFT_OBJ_ORIGIN_Z = (
  OBJECT_SUPPORT_Z
  + OBJECT_SURFACE_CLEARANCE
  - LEFT_COLLISION_ZMIN
  + OBJECT_INITIAL_HEIGHT_OFFSET
)
RIGHT_OBJ_ORIGIN_Z = (
  OBJECT_SUPPORT_Z
  + OBJECT_SURFACE_CLEARANCE
  - RIGHT_COLLISION_ZMIN
  + OBJECT_INITIAL_HEIGHT_OFFSET
)

# 物体初始位置
HALF_PI = 1.5707963267948966 # 物体旋转
LEFT_OBJ_INITIAL_POSE = {
  "x": SPAWN_REGION_X_RANGE,
  "y": LEFT_OBJ_SPAWN_Y_RANGE,
  "z": (LEFT_OBJ_ORIGIN_Z, LEFT_OBJ_ORIGIN_Z),
  "roll": (HALF_PI, HALF_PI),
  "pitch": (0.0, 0.0),
  "yaw": (HALF_PI, HALF_PI),
}
RIGHT_OBJ_INITIAL_POSE = {
  "x": SPAWN_REGION_X_RANGE,
  "y": RIGHT_OBJ_SPAWN_Y_RANGE,
  "z": (RIGHT_OBJ_ORIGIN_Z, RIGHT_OBJ_ORIGIN_Z),
  "roll": (HALF_PI, HALF_PI),
  "pitch": (0.0, 0.0),
  "yaw": (HALF_PI, HALF_PI),
}
OBJ_INITIAL_QUAT = (0.5, 0.5, 0.5, 0.5)
OBJECT_ROTATION_FAILURE_ANGLE = math.radians(45.0)  # 物体旋转失败角度

_LEFT_SCAN_URDF: Path = SRC_PATH / "assets" / "robots" / "left_obj" / "left_scan.urdf"
_RIGHT_SCAN_URDF: Path = SRC_PATH / "assets" / "robots" / "right_obj" / "right_scan.urdf"
_LEFT_SCAN_SDF: Path = SRC_PATH / "assets" / "robots" / "left_obj" / "left_scan_sdf.npz"
_RIGHT_SCAN_SDF: Path = SRC_PATH / "assets" / "robots" / "right_obj" / "right_scan_sdf.npz"
assert _LEFT_SCAN_URDF.is_file(), f"Missing left URDF: {_LEFT_SCAN_URDF}"
assert _RIGHT_SCAN_URDF.is_file(), f"Missing right URDF: {_RIGHT_SCAN_URDF}"


def _spec_from_scan_urdf(
  urdf_path: Path,
  initial_pos: tuple[float, float, float],
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(urdf_path))
  for body in spec.worldbody.bodies:
    if body.name == "base_link":
      body.pos[:] = initial_pos
      body.quat[:] = OBJ_INITIAL_QUAT
      body.add_freejoint(name=f"{urdf_path.stem}_free")
      return spec
  raise RuntimeError(f"Expected a 'base_link' body in {urdf_path}")


def _configure_object_hand_collision_filters(spec: mujoco.MjSpec) -> None:
  terrain_body = spec.body("terrain")
  for geom in terrain_body.geoms:
    geom.contype |= _OBJECT_COLLISION_BIT
    geom.conaffinity |= _OBJECT_COLLISION_BIT

  if OBJECT_HAND_COLLISIONS_ENABLED:
    return

  for body in spec.worldbody.bodies:
    if body.name not in {"left_obj/base_link", "right_obj/base_link"}:
      continue
    for geom in body.geoms:
      geom.contype = _OBJECT_COLLISION_BIT
      geom.conaffinity = _OBJECT_COLLISION_BIT


def _add_target_contact_sites(
  spec: mujoco.MjSpec,
  urdf_path: Path,
  sites: tuple[
    tuple[str, tuple[float, float, float], tuple[float, float, float, float]],
    ...,
  ],
) -> None:
  existing_site_names = {site.name for site in spec.sites}
  for body in spec.worldbody.bodies:
    if body.name == "base_link":
      for name, pos, rgba in sites:
        if name not in existing_site_names:
          body.add_site(
            name=name,
            pos=pos,
            size=(TARGET_HAND_JOINT_SITE_SIZE,),
            rgba=rgba,
          )
          existing_site_names.add(name)
      return
  raise RuntimeError(f"Expected a 'base_link' body in {urdf_path}")


def _add_position_site(
  spec: mujoco.MjSpec,
  urdf_path: Path,
  site: tuple[str, tuple[float, float, float], tuple[float, float, float, float]],
) -> None:
  name, pos, rgba = site
  existing_site_names = {s.name for s in spec.sites}
  if name in existing_site_names:
    return
  for body in spec.worldbody.bodies:
    if body.name == "base_link":
      body.add_site(
        name=name,
        pos=pos,
        size=(0.006,),
        rgba=rgba,
      )
      return
  raise RuntimeError(f"Expected a 'base_link' body in {urdf_path}")


def get_left_obj_spec() -> mujoco.MjSpec:
  spec = _spec_from_scan_urdf(
    _LEFT_SCAN_URDF,
    (
      (SPAWN_REGION_X_RANGE[0] + SPAWN_REGION_X_RANGE[1]) / 2.0,
      (LEFT_OBJ_SPAWN_Y_RANGE[0] + LEFT_OBJ_SPAWN_Y_RANGE[1]) / 2.0,
      LEFT_OBJ_ORIGIN_Z,
    ),
  )
  _add_target_contact_sites(spec, _LEFT_SCAN_URDF, LEFT_OBJ_LOCAL_TARGET_CONTACT_SITES)
  _add_target_contact_sites(spec, _LEFT_SCAN_URDF, LEFT_OBJ_LOCAL_TARGET_JOINT_SITES)
  _add_position_site(spec, _LEFT_SCAN_URDF, LEFT_OBJ_LOCAL_WRIST_TARGET_SITE)
  _add_position_site(spec, _LEFT_SCAN_URDF, RIGHT_OBJ_LOCAL_POSITION_TARGET1)
  _add_position_site(spec, _LEFT_SCAN_URDF, RIGHT_OBJ_LOCAL_POSITION_TARGET2)
  _add_position_site(spec, _LEFT_SCAN_URDF, RIGHT_OBJ_LOCAL_POSITION_TARGET3)
  return spec


def get_right_obj_spec() -> mujoco.MjSpec:
  spec = _spec_from_scan_urdf(
    _RIGHT_SCAN_URDF,
    (
      (SPAWN_REGION_X_RANGE[0] + SPAWN_REGION_X_RANGE[1]) / 2.0,
      (RIGHT_OBJ_SPAWN_Y_RANGE[0] + RIGHT_OBJ_SPAWN_Y_RANGE[1]) / 2.0,
      RIGHT_OBJ_ORIGIN_Z,
    ),
  )
  _add_target_contact_sites(spec, _RIGHT_SCAN_URDF, RIGHT_OBJ_LOCAL_TARGET_CONTACT_SITES)
  _add_target_contact_sites(spec, _RIGHT_SCAN_URDF, RIGHT_OBJ_LOCAL_TARGET_JOINT_SITES)
  _add_position_site(spec, _RIGHT_SCAN_URDF, RIGHT_OBJ_LOCAL_WRIST_TARGET_SITE)
  _add_position_site(spec, _RIGHT_SCAN_URDF, RIGHT_OBJ_LOCAL_POSITION_BEGIN)
  return spec


def _iter_env_origins(
  spec: mujoco.MjSpec,
  limit: int | None = None,
) -> list[tuple[int, float, float]]:
  env_sites = [s for s in spec.sites if s.name.startswith("env_origin_")]
  if not env_sites:
    return [(0, 0.0, 0.0)]
  if limit is not None:
    env_sites = env_sites[:limit]
  return [
    (env_id, float(site.pos[0]), float(site.pos[1]))
    for env_id, site in enumerate(env_sites)
  ]


def _add_spawn_rect_visual(
  spec: mujoco.MjSpec,
  env_id: int,
  origin_x: float,
  origin_y: float,
  ground_z: float,
) -> None:
  body = spec.worldbody.add_body(
    name=f"spawn_rect_env_{env_id}",
    pos=(origin_x, origin_y, ground_z),
  )
  rect_half_x = SPAWN_RECT_SIZE_X / 2.0
  rect_half_y = SPAWN_RECT_SIZE_Y / 2.0
  line_half = SPAWN_RECT_LINE_THICKNESS / 2.0
  rectangle_lines = (
    (
      "front",
      (SPAWN_RECT_CENTER_X, SPAWN_RECT_CENTER_Y + rect_half_y, SPAWN_RECT_LINE_HALF_Z),
      (rect_half_x, line_half, SPAWN_RECT_LINE_HALF_Z),
    ),
    (
      "back",
      (SPAWN_RECT_CENTER_X, SPAWN_RECT_CENTER_Y - rect_half_y, SPAWN_RECT_LINE_HALF_Z),
      (rect_half_x, line_half, SPAWN_RECT_LINE_HALF_Z),
    ),
    (
      "left",
      (SPAWN_RECT_CENTER_X - rect_half_x, SPAWN_RECT_CENTER_Y, SPAWN_RECT_LINE_HALF_Z),
      (line_half, rect_half_y, SPAWN_RECT_LINE_HALF_Z),
    ),
    (
      "right",
      (SPAWN_RECT_CENTER_X + rect_half_x, SPAWN_RECT_CENTER_Y, SPAWN_RECT_LINE_HALF_Z),
      (line_half, rect_half_y, SPAWN_RECT_LINE_HALF_Z),
    ),
  )
  for line_name, line_pos, line_size in rectangle_lines:
    body.add_geom(
      name=f"spawn_rect_{line_name}_env_{env_id}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=line_pos,
      size=line_size,
      rgba=SPAWN_RECT_RGBA,
      contype=0,
      conaffinity=0,
    )


def _add_spawn_rect_visuals(spec: mujoco.MjSpec) -> None:
  for env_id, ox, oy in _iter_env_origins(spec, limit=SPAWN_RECT_VISUAL_ENV_LIMIT):
    _add_spawn_rect_visual(spec, env_id, ox, oy, ground_z=OBJECT_SUPPORT_Z)


def _raise_terrain_to_support_height(spec: mujoco.MjSpec) -> None:
  terrain_body = spec.body("terrain")
  terrain_body.pos[2] = OBJECT_SUPPORT_Z


def _set_site_visualization(spec: mujoco.MjSpec, *, visible: bool) -> None:
  if visible:
    return
  for site in spec.sites:
    site.rgba[3] = 0.0


def _configure_train_scene_spec(spec: mujoco.MjSpec) -> None:
  _raise_terrain_to_support_height(spec)
  _configure_object_hand_collision_filters(spec)
  _set_site_visualization(spec, visible=False)


def _configure_play_scene_spec(spec: mujoco.MjSpec) -> None:
  _raise_terrain_to_support_height(spec)
  _configure_object_hand_collision_filters(spec)
  _add_spawn_rect_visuals(spec)


def configure_lift_cube_scene(
  cfg: ManagerBasedRlEnvCfg,
  robot_cfg: EntityCfg,
  *,
  show_spawn_rect: bool = False,
) -> None:
  cfg.scene.entities = {
    "robot": robot_cfg,
    "left_obj": EntityCfg(spec_fn=get_left_obj_spec),
    "right_obj": EntityCfg(spec_fn=get_right_obj_spec),
  }
  cfg.scene.env_spacing = 1.5
  cfg.scene.spec_fn = (
    _configure_play_scene_spec if show_spawn_rect else _configure_train_scene_spec
  )

  cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
  cfg.viewer.entity_name = None
  cfg.viewer.body_name = None
  cfg.viewer.lookat = (SPAWN_RECT_CENTER_X, SPAWN_RECT_CENTER_Y, OBJECT_SUPPORT_Z)


@dataclass(frozen=True)
class ObjectResetEventSettings:
  name: str
  entity_name: str
  pose_range: Mapping[str, tuple[float, float]]
  velocity_range: Mapping[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class ContactSensorSettings:
  sensor_name: str
  primary_pattern: str | tuple[str, ...]
  object_name: str | None = None
  primary_mode: str = "geom"
  primary_entity: str = "robot"
  secondary_mode: str = "body"
  secondary_pattern: str | tuple[str, ...] = "base_link"
  secondary_entity: str | None = None


@dataclass(frozen=True)
class LiftCubeManagerSettings:
  observation_nan_policy: str | None = None
  observation_clips: Mapping[str, tuple[float, float]] = field(default_factory=dict)
  action_scale: float | None = None
  object_reset_events: tuple[ObjectResetEventSettings, ...] = ()
  fingertip_friction_geom_names: str | tuple[str, ...] | None = None
  fingertip_object_contact_sensors: tuple[ContactSensorSettings, ...] = ()
  palm_ground_contact_sensors: tuple[ContactSensorSettings, ...] = ()
  selected_fingertip_contact_sensor_names: tuple[str, ...] = ()
  invalid_physics_termination_params: Mapping[str, Any] | None = None
  object_rotation_termination_params: Mapping[str, Any] | None = None


def _contact_match(
  mode: str,
  pattern: str | tuple[str, ...],
  entity: str | None = None,
) -> ContactMatch:
  kwargs: dict[str, Any] = {"mode": mode, "pattern": pattern}
  if entity is not None:
    kwargs["entity"] = entity
  return ContactMatch(**kwargs)


def _make_contact_sensor(
  settings: ContactSensorSettings,
) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=settings.sensor_name,
    primary=_contact_match(
      settings.primary_mode,
      settings.primary_pattern,
      settings.primary_entity,
    ),
    secondary=_contact_match(
      settings.secondary_mode,
      settings.secondary_pattern,
      settings.secondary_entity if settings.secondary_entity is not None else settings.object_name,
    ),
    fields=("found", "force", "normal", "tangent"),
    reduce="none",
    num_slots=1,
    global_frame=True,
  )


def _make_default_manager_settings() -> LiftCubeManagerSettings:
  hand_geom_pattern = r"^(left|right)_(palm|finger).*"
  return LiftCubeManagerSettings(
    observation_nan_policy="sanitize",
    observation_clips={
      "joint_pos": (-20.0, 20.0),
      "joint_vel": (-100.0, 100.0),
      "object_positions": (-5.0, 5.0),
      "object_quaternions": (-1.0, 1.0),
      "fingertip_contact_forces": (-20.0, 20.0),
      "right_object_to_fingertips": (-5.0, 5.0),
      "left_object_to_fingertips": (-5.0, 5.0),
      "left_fingertip_to_target_sites": (-5.0, 5.0),
      "right_fingertip_to_target_sites": (-5.0, 5.0),
      "right_object_stage_targets": (-5.0, 5.0),
      "right_object_stage_state": (0.0, 5.0),
      "actions": (-20.0, 20.0),
    },
    action_scale=G1_WUJI_MANIP_ACTION_SCALE,
    object_reset_events=(
      ObjectResetEventSettings(
        name="reset_left_obj",
        entity_name="left_obj",
        pose_range=LEFT_OBJ_INITIAL_POSE,
      ),
      ObjectResetEventSettings(
        name="reset_right_obj",
        entity_name="right_obj",
        pose_range=RIGHT_OBJ_INITIAL_POSE,
      ),
    ),
    fingertip_friction_geom_names=hand_geom_pattern,
    fingertip_object_contact_sensors=(
      ContactSensorSettings(
        sensor_name="right_fingertip_right_obj_contact",
        primary_pattern=(
          "right_finger1_tip_link_collision",
          "right_finger2_tip_link_collision",
          "right_finger3_tip_link_collision",
          "right_finger4_tip_link_collision",
          "right_finger5_tip_link_collision",
        ),
        object_name="right_obj",
      ),
      ContactSensorSettings(
        sensor_name="left_fingertip_left_obj_contact",
        primary_pattern=(
          "left_finger1_tip_link_collision",
          "left_finger2_tip_link_collision",
          "left_finger3_tip_link_collision",
          "left_finger4_tip_link_collision",
          "left_finger5_tip_link_collision",
        ),
        object_name="left_obj",
      ),
    ),
    palm_ground_contact_sensors=(
      ContactSensorSettings(
        sensor_name="left_palm_ground_contact",
        primary_pattern="left_palm_link_collision",
        secondary_pattern="terrain",
      ),
      ContactSensorSettings(
        sensor_name="right_palm_ground_contact",
        primary_pattern="right_palm_link_collision",
        secondary_pattern="terrain",
      ),
    ),
    invalid_physics_termination_params={
      "asset_names": ("robot", "left_obj", "right_obj"),
      "max_joint_pos_abs": 20.0,
      "max_joint_vel_abs": 100.0,
      "max_root_distance": 5.0,
      "max_root_vel_abs": 50.0,
      "max_body_distance": 5.0,
    },
    object_rotation_termination_params={
      "asset_names": ("left_obj", "right_obj"),
      "max_angle": OBJECT_ROTATION_FAILURE_ANGLE,
      "reference_euler_xyz_by_asset": {
        "left_obj": (HALF_PI, 0.0, HALF_PI),
        "right_obj": (HALF_PI, 0.0, HALF_PI),
      },
    },
  )


def make_lift_cube_env_cfg(
  *,
  play: bool = False,
  manager_settings: LiftCubeManagerSettings | None = None,
  reward_mode: str = "approach",
) -> ManagerBasedRlEnvCfg:
  """Create base cube lifting task configuration."""
  manager_settings = manager_settings or _make_default_manager_settings()
  if reward_mode not in {"approach", "contact", "move"}:
    raise ValueError(f"Unsupported reward_mode: {reward_mode}")
  is_residual_reward_mode = reward_mode in {"contact", "move"}
  is_move_reward_mode = reward_mode == "move"

  left_fingertip_observation_cfg = SceneEntityCfg(
    "robot",
    site_names=tuple(f"left_fingertip_site_{s}" for s in FINGER_SITE_SUFFIXES),
    preserve_order=True,
  )
  right_fingertip_observation_cfg = SceneEntityCfg(
    "robot",
    site_names=tuple(f"right_fingertip_site_{s}" for s in FINGER_SITE_SUFFIXES),
    preserve_order=True,
  )
  fingertip_contact_sensor_names = (
    manager_settings.selected_fingertip_contact_sensor_names
    or tuple(
      sensor.sensor_name
      for sensor in manager_settings.fingertip_object_contact_sensors
    )
  )
  contact_sensor_names_by_object = {
    sensor.object_name: sensor.sensor_name
    for sensor in manager_settings.fingertip_object_contact_sensors
  }
  palm_sensor_names_by_side = {
    side: tuple(
      sensor.sensor_name
      for sensor in manager_settings.palm_ground_contact_sensors
      if sensor.sensor_name.startswith(f"{side}_")
    )
    for side in ("left", "right")
  }
  for side, sensor_names in palm_sensor_names_by_side.items():
    if not sensor_names:
      raise KeyError(f"Missing palm-ground contact sensor for {side} hand.")

  def make_right_object_stage_shared_params() -> dict[str, Any]:
    return {
      "left_fingertip_cfg": SceneEntityCfg(
        "robot",
        site_names=("left_fingertip_site_t", "left_fingertip_site_i"),
        preserve_order=True,
      ),
      "left_target_cfg": SceneEntityCfg(
        "left_obj",
        site_names=("left_target_contact_site_t", "left_target_contact_site_i"),
        preserve_order=True,
      ),
      "right_fingertip_cfg": SceneEntityCfg(
        "robot",
        site_names=(
          "right_fingertip_site_t",
          "right_fingertip_site_i",
          "right_fingertip_site_m",
        ),
        preserve_order=True,
      ),
      "right_target_cfg": SceneEntityCfg(
        "right_obj",
        site_names=(
          "right_target_contact_site_t",
          "right_target_contact_site_i",
          "right_target_contact_site_m",
        ),
        preserve_order=True,
      ),
      "moving_site_cfg": SceneEntityCfg(
        "right_obj",
        site_names=(RIGHT_OBJ_LOCAL_POSITION_BEGIN[0],),
        preserve_order=True,
      ),
      "stage_target_cfg": SceneEntityCfg(
        "left_obj",
        site_names=(
          RIGHT_OBJ_LOCAL_POSITION_TARGET1[0],
          RIGHT_OBJ_LOCAL_POSITION_TARGET2[0],
          RIGHT_OBJ_LOCAL_POSITION_TARGET3[0],
        ),
        preserve_order=True,
      ),
      "activation_threshold": RIGHT_OBJECT_STAGE_ACTIVATION_THRESHOLD,
      "activation_force_threshold": FINGERTIP_OBJECT_CONTACT_FORCE_THRESHOLD,
      "left_sensor_name": contact_sensor_names_by_object["left_obj"],
      "right_sensor_name": contact_sensor_names_by_object["right_obj"],
      "activation_hold_steps": RIGHT_OBJECT_STAGE_ACTIVATION_HOLD_STEPS,
      "stage_thresholds": RIGHT_OBJECT_STAGE_THRESHOLDS,
      "stage_exp_scales": RIGHT_OBJECT_STAGE_EXP_SCALES,
    }

  def make_right_object_target3_curriculum_params() -> dict[str, Any]:
    return {
      "targets": (
        {
          "target_entity_name": "left_obj",
          "target_site_name": RIGHT_OBJ_LOCAL_POSITION_TARGET2[0],
          "start_y": RIGHT_OBJECT_TARGET2_CURRICULUM_START_Y,
          "end_y": RIGHT_OBJECT_TARGET2_CURRICULUM_END_Y,
          "log_key": "target2_y",
        },
        {
          "target_entity_name": "left_obj",
          "target_site_name": RIGHT_OBJ_LOCAL_POSITION_TARGET3[0],
          "start_y": RIGHT_OBJECT_TARGET3_CURRICULUM_START_Y,
          "end_y": RIGHT_OBJECT_TARGET3_CURRICULUM_END_Y,
          "log_key": "target3_y",
        },
      ),
      "success_threshold": RIGHT_OBJECT_TARGET_CURRICULUM_SUCCESS_THRESHOLD,
      "required_consecutive": RIGHT_OBJECT_TARGET_CURRICULUM_REQUIRED_CONSECUTIVE,
      "difficulty_step": RIGHT_OBJECT_TARGET_CURRICULUM_DIFFICULTY_STEP,
      "max_difficulty": 1.0,
    }

  def make_right_object_target_play_difficulty_params() -> dict[str, Any]:
    return {
      "targets": make_right_object_target3_curriculum_params()["targets"],
      "difficulty": RIGHT_OBJECT_TARGET_PLAY_DIFFICULTY,
    }

  def make_observation_terms(*, add_noise: bool) -> dict[str, ObservationTermCfg]:
    return {
      "joint_pos": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        noise=Unoise(*OBS_JOINT_POS_NOISE_RANGE) if add_noise else None,
      ),
      "joint_vel": ObservationTermCfg(
        func=mdp.joint_vel_rel,
        noise=Unoise(*OBS_JOINT_VEL_NOISE_RANGE) if add_noise else None,
      ),
      "object_positions": ObservationTermCfg(
        func=manipulation_mdp.object_positions,
        params={"object_names": ("left_obj", "right_obj")},
        noise=Unoise(*OBS_POSITION_NOISE_RANGE) if add_noise else None,
      ),
      "object_quaternions": ObservationTermCfg(
        func=manipulation_mdp.object_quaternions,
        params={"object_names": ("left_obj", "right_obj")},
        noise=Unoise(*OBS_QUATERNION_NOISE_RANGE) if add_noise else None,
      ),
      "fingertip_contact_forces": ObservationTermCfg(
        func=manipulation_mdp.fingertip_contact_forces,
        params={
          "sensor_name": fingertip_contact_sensor_names,
          "include_magnitude": True,
          "log_scale": True,
        },
        noise=Unoise(*OBS_CONTACT_FORCE_NOISE_RANGE) if add_noise else None,
      ),
      "right_object_to_fingertips": ObservationTermCfg(
        func=manipulation_mdp.object_to_fingertip_distance,
        params={
          "object_name": "right_obj",
          "fingertip_cfg": right_fingertip_observation_cfg,
          "include_distance": True,
        },
        noise=Unoise(*OBS_POSITION_NOISE_RANGE) if add_noise else None,
      ),
      "left_object_to_fingertips": ObservationTermCfg(
        func=manipulation_mdp.object_to_fingertip_distance,
        params={
          "object_name": "left_obj",
          "fingertip_cfg": left_fingertip_observation_cfg,
          "include_distance": True,
        },
        noise=Unoise(*OBS_POSITION_NOISE_RANGE) if add_noise else None,
      ),
      "left_fingertip_to_target_sites": ObservationTermCfg(
        func=manipulation_mdp.site_to_site_relative,
        params={
          "source_cfg": SceneEntityCfg(
            "robot",
            site_names=tuple(f"left_fingertip_site_{s}" for s in FINGER_SITE_SUFFIXES),
            preserve_order=True,
          ),
          "target_cfg": SceneEntityCfg(
            "left_obj",
            site_names=tuple(
              f"left_target_contact_site_{s}" for s in FINGER_SITE_SUFFIXES
            ),
            preserve_order=True,
          ),
          "include_distance": True,
        },
        noise=Unoise(*OBS_POSITION_NOISE_RANGE) if add_noise else None,
      ),
      "right_fingertip_to_target_sites": ObservationTermCfg(
        func=manipulation_mdp.site_to_site_relative,
        params={
          "source_cfg": SceneEntityCfg(
            "robot",
            site_names=tuple(f"right_fingertip_site_{s}" for s in FINGER_SITE_SUFFIXES),
            preserve_order=True,
          ),
          "target_cfg": SceneEntityCfg(
            "right_obj",
            site_names=tuple(
              f"right_target_contact_site_{s}" for s in FINGER_SITE_SUFFIXES
            ),
            preserve_order=True,
          ),
          "include_distance": True,
        },
        noise=Unoise(*OBS_POSITION_NOISE_RANGE) if add_noise else None,
      ),
      "right_object_stage_targets": ObservationTermCfg(
        func=manipulation_mdp.right_object_stage_target_relative,
        params={
          "moving_site_cfg": SceneEntityCfg(
            "right_obj",
            site_names=(RIGHT_OBJ_LOCAL_POSITION_BEGIN[0],),
            preserve_order=True,
          ),
          "target_site_cfg": SceneEntityCfg(
            "left_obj",
            site_names=(
              RIGHT_OBJ_LOCAL_POSITION_TARGET1[0],
              RIGHT_OBJ_LOCAL_POSITION_TARGET2[0],
              RIGHT_OBJ_LOCAL_POSITION_TARGET3[0],
            ),
            preserve_order=True,
          ),
          "include_distance": True,
        },
        noise=Unoise(*OBS_POSITION_NOISE_RANGE) if add_noise else None,
      ),
      "right_object_stage_state": ObservationTermCfg(
        func=manipulation_mdp.right_object_stage_state,
        params=make_right_object_stage_shared_params(),
      ),
      "actions": ObservationTermCfg(func=mdp.last_action),
    }

  actor_terms = make_observation_terms(add_noise=True)
  critic_terms = make_observation_terms(add_noise=False)
  for terms in (actor_terms, critic_terms):
    for term_name, clip in manager_settings.observation_clips.items():
      if term_name in terms:
        terms[term_name].clip = clip

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }
  if manager_settings.observation_nan_policy is not None:
    for obs_group in observations.values():
      obs_group.nan_policy = manager_settings.observation_nan_policy

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=manager_settings.action_scale
      if manager_settings.action_scale is not None
      else 0.5,
      use_default_offset=True,
    )
  }

  events = {
    # For positioning the base of the robot at env_origins.
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": INITIAL_JOINT_POSITION_RANDOM_RANGE,
        "velocity_range": INITIAL_JOINT_VELOCITY_RANDOM_RANGE,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "randomize_robot_pd_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "kp_range": PD_GAIN_RANDOM_SCALE_RANGE,
        "kd_range": PD_GAIN_RANDOM_SCALE_RANGE,
        "asset_cfg": SceneEntityCfg("robot"),
        "distribution": "uniform",
        "operation": "scale",
      },
    ),
    "fingertip_friction_slide": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0],
        "ranges": (0.3, 1.5),
      },
    ),
    "fingertip_friction_spin": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "distribution": "log_uniform",
        "axes": [1],
        "ranges": (1e-4, 2e-2),
      },
    ),
    "fingertip_friction_roll": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "distribution": "log_uniform",
        "axes": [2],
        "ranges": (1e-5, 5e-3),
      },
    ),
  }
  for reset_event in manager_settings.object_reset_events:
    events[reset_event.name] = EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": dict(reset_event.pose_range),
        "velocity_range": dict(reset_event.velocity_range),
        "asset_cfg": SceneEntityCfg(reset_event.entity_name),
      },
    )
  if is_move_reward_mode and play:
    events["set_right_object_target_play_difficulty"] = EventTermCfg(
      mode="startup",
      func=manipulation_mdp.set_right_object_target_difficulty,
      params=make_right_object_target_play_difficulty_params(),
    )

  if manager_settings.fingertip_friction_geom_names is not None:
    for event_name in (
      "fingertip_friction_slide",
      "fingertip_friction_spin",
      "fingertip_friction_roll",
    ):
      events[event_name].params["asset_cfg"].geom_names = (
        manager_settings.fingertip_friction_geom_names
      )

  scene_sensors = tuple(
    _make_contact_sensor(sensor_settings)
    for sensor_settings in (
      *manager_settings.fingertip_object_contact_sensors,
      *manager_settings.palm_ground_contact_sensors,
    )
  )

  def make_fingertip_wrist_alignment_reward_params() -> dict[str, Any]:
    # Return fresh SceneEntityCfg instances each call to avoid resolve-state reuse.
    return {
      "left_fingertip_cfg": SceneEntityCfg(
        "robot",
        site_names=tuple(f"left_fingertip_site_{s}" for s in FINGER_SITE_SUFFIXES),
        preserve_order=True,
      ),
      "left_target_cfg": SceneEntityCfg(
        "left_obj",
        site_names=tuple(f"left_target_contact_site_{s}" for s in FINGER_SITE_SUFFIXES),
        preserve_order=True,
      ),
      "right_fingertip_cfg": SceneEntityCfg(
        "robot",
        site_names=tuple(f"right_fingertip_site_{s}" for s in FINGER_SITE_SUFFIXES),
        preserve_order=True,
      ),
      "right_target_cfg": SceneEntityCfg(
        "right_obj",
        site_names=tuple(f"right_target_contact_site_{s}" for s in FINGER_SITE_SUFFIXES),
        preserve_order=True,
      ),
      "left_wrist_cfg": SceneEntityCfg(
        "robot",
        site_names=("left_wrist_site",),
        preserve_order=True,
      ),
      "left_wrist_target_cfg": SceneEntityCfg(
        "left_obj",
        site_names=("left_wrist_target_site",),
        preserve_order=True,
      ),
      "right_wrist_cfg": SceneEntityCfg(
        "robot",
        site_names=("right_wrist_site",),
        preserve_order=True,
      ),
      "right_wrist_target_cfg": SceneEntityCfg(
        "right_obj",
        site_names=("right_wrist_target_site",),
        preserve_order=True,
      ),
      "left_scales": FINGERTIP_TARGET_DISTANCE_SCALES,
      "right_scales": FINGERTIP_TARGET_DISTANCE_SCALES,
      "left_wrist_scale": WRIST_TARGET_DISTANCE_SCALE,
      "right_wrist_scale": WRIST_TARGET_DISTANCE_SCALE,
    }

  def make_contact_graph_reward_params() -> dict[str, Any]:
    return {
      "left_fingertip_cfg": SceneEntityCfg(
        "robot",
        site_names=tuple(f"left_fingertip_site_{s}" for s in FINGER_SITE_SUFFIXES),
        preserve_order=True,
      ),
      "left_target_cfg": SceneEntityCfg(
        "left_obj",
        site_names=tuple(f"left_target_contact_site_{s}" for s in FINGER_SITE_SUFFIXES),
        preserve_order=True,
      ),
      "right_fingertip_cfg": SceneEntityCfg(
        "robot",
        site_names=tuple(f"right_fingertip_site_{s}" for s in FINGER_SITE_SUFFIXES),
        preserve_order=True,
      ),
      "right_target_cfg": SceneEntityCfg(
        "right_obj",
        site_names=tuple(f"right_target_contact_site_{s}" for s in FINGER_SITE_SUFFIXES),
        preserve_order=True,
      ),
      "left_sensor_name": contact_sensor_names_by_object["left_obj"],
      "right_sensor_name": contact_sensor_names_by_object["right_obj"],
      "left_target_graph": LEFT_TARGET_CONTACT_GRAPH,
      "right_target_graph": RIGHT_TARGET_CONTACT_GRAPH,
      "distance_threshold": CONTACT_GRAPH_DISTANCE_THRESHOLD,
      "force_threshold": FINGERTIP_OBJECT_CONTACT_FORCE_THRESHOLD,
    }

  def make_single_side_contact_graph_reward_params(hand: str) -> dict[str, Any]:
    object_name = f"{hand}_obj"
    target_graph = (
      LEFT_TARGET_CONTACT_GRAPH if hand == "left" else RIGHT_TARGET_CONTACT_GRAPH
    )
    return {
      "fingertip_cfg": SceneEntityCfg(
        "robot",
        site_names=tuple(f"{hand}_fingertip_site_{s}" for s in FINGER_SITE_SUFFIXES),
        preserve_order=True,
      ),
      "target_cfg": SceneEntityCfg(
        object_name,
        site_names=tuple(f"{hand}_target_contact_site_{s}" for s in FINGER_SITE_SUFFIXES),
        preserve_order=True,
      ),
      "sensor_name": contact_sensor_names_by_object[object_name],
      "target_graph": target_graph,
      "distance_threshold": CONTACT_GRAPH_DISTANCE_THRESHOLD,
      "force_threshold": FINGERTIP_OBJECT_CONTACT_FORCE_THRESHOLD,
      "exp_scale": CONTACT_GRAPH_EXP_SCALE,
    }

  def make_single_side_key_fingertip_force_reward_params(hand: str) -> dict[str, Any]:
    object_name = f"{hand}_obj"
    target_graph = (
      LEFT_TARGET_CONTACT_GRAPH if hand == "left" else RIGHT_TARGET_CONTACT_GRAPH
    )
    return {
      "sensor_name": contact_sensor_names_by_object[object_name],
      "target_graph": target_graph,
      "force_scale": KEY_FINGERTIP_FORCE_EXP_SCALE,
    }

  def make_palm_ground_contact_penalty_params() -> dict[str, Any]:
    return {
      "sensor_names": tuple(
        sensor.sensor_name for sensor in manager_settings.palm_ground_contact_sensors
      ),
    }

  def make_right_object_stage_params() -> dict[str, Any]:
    params = make_right_object_stage_shared_params()
    params["stage_interp_counts"] = RIGHT_OBJECT_STAGE_INTERP_COUNTS
    params["segment_exp_scale"] = RIGHT_OBJECT_STAGE_SEGMENT_EXP_SCALE
    params["height_exp_scale"] = RIGHT_OBJECT_STAGE_HEIGHT_EXP_SCALE
    params["height_progress_mix"] = RIGHT_OBJECT_STAGE_HEIGHT_PROGRESS_MIX
    params["height_track_mix"] = RIGHT_OBJECT_STAGE_HEIGHT_TRACK_MIX
    params["segment_reward_weight"] = RIGHT_OBJECT_STAGE_SEGMENT_REWARD_WEIGHT
    params["waypoint_reward_weight"] = RIGHT_OBJECT_STAGE_WAYPOINT_REWARD_WEIGHT
    params["height_reward_weight"] = RIGHT_OBJECT_STAGE_HEIGHT_REWARD_WEIGHT
    params["passed_waypoint_reward_weight"] = (
      RIGHT_OBJECT_STAGE_PASSED_WAYPOINT_REWARD_WEIGHT
    )
    params["success_reward_multiplier"] = (
      RIGHT_OBJECT_STAGE_SUCCESS_REWARD_MULTIPLIER
    )
    return params

  def make_right_object_stage_no_progress_params() -> dict[str, Any]:
    params = make_right_object_stage_shared_params()
    params["stage_interp_counts"] = RIGHT_OBJECT_STAGE_INTERP_COUNTS
    params["no_progress_steps"] = RIGHT_OBJECT_STAGE_NO_PROGRESS_TERMINATION_STEPS
    return params

  def make_right_object_stage_success_params() -> dict[str, Any]:
    params = make_right_object_stage_shared_params()
    params["stage_interp_counts"] = RIGHT_OBJECT_STAGE_INTERP_COUNTS
    return params

  def make_contact_move_object_regularization_params() -> dict[str, Any]:
    stage_params = make_right_object_stage_shared_params()
    return {
      "left_fingertip_cfg": stage_params["left_fingertip_cfg"],
      "left_target_cfg": stage_params["left_target_cfg"],
      "right_fingertip_cfg": stage_params["right_fingertip_cfg"],
      "right_target_cfg": stage_params["right_target_cfg"],
      "left_sensor_name": stage_params["left_sensor_name"],
      "right_sensor_name": stage_params["right_sensor_name"],
      "activation_threshold": stage_params["activation_threshold"],
      "activation_force_threshold": stage_params["activation_force_threshold"],
      "activation_hold_steps": stage_params["activation_hold_steps"],
      "allow_right_object_motion_after_grasp": is_move_reward_mode,
      "left_object_name": "left_obj",
      "right_object_name": "right_obj",
      "lift_tolerance": CONTACT_MOVE_OBJECT_LIFT_TOLERANCE,
      "linear_velocity_tolerance": CONTACT_MOVE_OBJECT_LINEAR_VELOCITY_TOLERANCE,
      "angular_velocity_tolerance": CONTACT_MOVE_OBJECT_ANGULAR_VELOCITY_TOLERANCE,
      "left_xy_displacement_tolerance": LEFT_OBJECT_XY_DISPLACEMENT_TOLERANCE,
      "right_xy_displacement_tolerance": RIGHT_OBJECT_XY_DISPLACEMENT_TOLERANCE,
      "left_linear_velocity_weight": LEFT_OBJECT_LINEAR_VELOCITY_PENALTY_WEIGHT,
      "left_angular_velocity_weight": LEFT_OBJECT_ANGULAR_VELOCITY_PENALTY_WEIGHT,
      "left_xy_displacement_weight": LEFT_OBJECT_XY_DISPLACEMENT_PENALTY_WEIGHT,
      "left_lift_weight": LEFT_OBJECT_LIFT_PENALTY_WEIGHT,
      "right_pre_grasp_linear_velocity_weight": (
        RIGHT_OBJECT_PRE_GRASP_LINEAR_VELOCITY_PENALTY_WEIGHT
      ),
      "right_pre_grasp_angular_velocity_weight": (
        RIGHT_OBJECT_PRE_GRASP_ANGULAR_VELOCITY_PENALTY_WEIGHT
      ),
      "right_xy_displacement_weight": RIGHT_OBJECT_XY_DISPLACEMENT_PENALTY_WEIGHT,
      "right_move_linear_velocity_limit": RIGHT_OBJECT_MOVE_LINEAR_VELOCITY_LIMIT,
      "right_move_angular_velocity_limit": RIGHT_OBJECT_MOVE_ANGULAR_VELOCITY_LIMIT,
      "right_move_linear_velocity_weight": (
        RIGHT_OBJECT_MOVE_LINEAR_VELOCITY_PENALTY_WEIGHT
      ),
      "right_move_angular_velocity_weight": (
        RIGHT_OBJECT_MOVE_ANGULAR_VELOCITY_PENALTY_WEIGHT
      ),
      "right_speed_weight": CONTACT_MOVE_RIGHT_SPEED_REWARD_WEIGHT,
      "right_lift_weight": RIGHT_OBJECT_LIFT_PENALTY_WEIGHT,
      "include_left": not is_move_reward_mode,
      "include_right": True,
    }

  def make_contact_move_relative_pose_params() -> dict[str, Any]:
    stage_params = make_right_object_stage_shared_params()
    left_key_suffixes = ("t", "i")
    right_key_suffixes = ("t", "i", "m")

    def key_hand_site_names(hand: str, suffixes: tuple[str, ...]) -> tuple[str, ...]:
      joint_site_names = tuple(
        f"{hand}_{joint_name}_site_{suffix}"
        for joint_name in TARGET_HAND_JOINT_NAMES
        for suffix in suffixes
      )
      fingertip_site_names = tuple(
        f"{hand}_fingertip_site_{suffix}" for suffix in suffixes
      )
      return joint_site_names + fingertip_site_names

    return {
      "left_fingertip_cfg": stage_params["left_fingertip_cfg"],
      "left_target_cfg": stage_params["left_target_cfg"],
      "right_fingertip_cfg": stage_params["right_fingertip_cfg"],
      "right_target_cfg": stage_params["right_target_cfg"],
      "left_sensor_name": stage_params["left_sensor_name"],
      "right_sensor_name": stage_params["right_sensor_name"],
      "activation_threshold": stage_params["activation_threshold"],
      "activation_force_threshold": stage_params["activation_force_threshold"],
      "activation_hold_steps": stage_params["activation_hold_steps"],
      "left_relative_site_cfg": SceneEntityCfg(
        "robot",
        site_names=key_hand_site_names("left", left_key_suffixes),
        preserve_order=True,
      ),
      "right_relative_site_cfg": SceneEntityCfg(
        "robot",
        site_names=key_hand_site_names("right", right_key_suffixes),
        preserve_order=True,
      ),
      "left_object_name": "left_obj",
      "right_object_name": "right_obj",
      "exp_scale": CONTACT_MOVE_RELATIVE_POSE_EXP_SCALE,
      "include_left": not is_move_reward_mode,
      "include_right": True,
    }

  def make_key_fingertip_target_distance_penalty_params() -> dict[str, Any]:
    stage_params = make_right_object_stage_shared_params()
    return {
      "left_fingertip_cfg": stage_params["left_fingertip_cfg"],
      "left_target_cfg": stage_params["left_target_cfg"],
      "right_fingertip_cfg": stage_params["right_fingertip_cfg"],
      "right_target_cfg": stage_params["right_target_cfg"],
      "distance_threshold": KEY_FINGERTIP_TARGET_DISTANCE_FAILURE_THRESHOLD,
      "include_left": not is_move_reward_mode,
      "include_right": True,
    }

  def make_key_fingertip_target_distance_termination_params() -> dict[str, Any]:
    stage_params = make_right_object_stage_shared_params()
    return {
      "left_fingertip_cfg": stage_params["left_fingertip_cfg"],
      "left_target_cfg": stage_params["left_target_cfg"],
      "right_fingertip_cfg": stage_params["right_fingertip_cfg"],
      "right_target_cfg": stage_params["right_target_cfg"],
      "distance_threshold": KEY_FINGERTIP_TARGET_DISTANCE_TERMINATION_THRESHOLD,
      "active_after_steps": (
        KEY_FINGERTIP_TARGET_DISTANCE_TERMINATION_ACTIVE_AFTER_STEPS
      ),
    }

  def make_single_hand_alignment_reward_params(hand: str) -> dict[str, Any]:
    object_name = f"{hand}_obj"
    object_sdf_file = _LEFT_SCAN_SDF if hand == "left" else _RIGHT_SCAN_SDF
    return {
      "fingertip_cfg": SceneEntityCfg(
        "robot",
        site_names=tuple(f"{hand}_fingertip_site_{s}" for s in FINGER_SITE_SUFFIXES),
        preserve_order=True,
      ),
      "target_cfg": SceneEntityCfg(
        object_name,
        site_names=tuple(f"{hand}_target_contact_site_{s}" for s in FINGER_SITE_SUFFIXES),
        preserve_order=True,
      ),
      "wrist_cfg": SceneEntityCfg(
        "robot",
        site_names=(f"{hand}_wrist_site",),
        preserve_order=True,
      ),
      "wrist_target_cfg": SceneEntityCfg(
        object_name,
        site_names=(f"{hand}_wrist_target_site",),
        preserve_order=True,
      ),
      "joint_site_cfg": SceneEntityCfg(
        "robot",
        site_names=tuple(
          f"{hand}_{joint_name}_site_{suffix}"
          for joint_name in TARGET_HAND_JOINT_NAMES
          for suffix in FINGER_SITE_SUFFIXES
        ),
        preserve_order=True,
      ),
      "joint_target_cfg": SceneEntityCfg(
        object_name,
        site_names=tuple(
          f"{hand}_target_{joint_name}_site_{suffix}"
          for joint_name in TARGET_HAND_JOINT_NAMES
          for suffix in FINGER_SITE_SUFFIXES
        ),
        preserve_order=True,
      ),
      "fingertip_scales": FINGERTIP_TARGET_DISTANCE_SCALES,
      "wrist_scale": WRIST_TARGET_DISTANCE_SCALE,
      "joint_site_scale": HAND_JOINT_TARGET_DISTANCE_SCALE,
      "joint_site_primary_reward_weight": HAND_JOINT_PRIMARY_FINGER_REWARD_WEIGHT,
      "joint_site_secondary_reward_weight": HAND_JOINT_SECONDARY_FINGER_REWARD_WEIGHT,
      "site_height_penalty_weight": HAND_SITE_HEIGHT_PENALTY_WEIGHT,
      "object_sdf_file": str(object_sdf_file),
      "object_sdf_name": object_name,
      "object_sdf_early_clearance": HAND_OBJECT_SDF_EARLY_CLEARANCE,
      "object_sdf_later_clearance": HAND_OBJECT_SDF_LATER_CLEARANCE,
      "object_sdf_early_penalty_weight": HAND_OBJECT_SDF_EARLY_PENALTY_WEIGHT,
      "object_sdf_later_penalty_weight": HAND_OBJECT_SDF_LATER_PENALTY_WEIGHT,
      "object_sdf_later_threshold": HAND_OBJECT_SDF_LATER_THRESHOLD,
      "object_sdf_samples_per_segment": HAND_OBJECT_SDF_SAMPLES_PER_SEGMENT,
      "object_sdf_early_step_window": HAND_OBJECT_SDF_EARLY_STEP_WINDOW,
      "object_sdf_early_penalty_multiplier": HAND_OBJECT_SDF_EARLY_PENALTY_MULTIPLIER,
    }

  def make_single_hand_internal_common_params(hand: str) -> dict[str, Any]:
    return {
      "core_reward_weight": HAND_INTERNAL_CORE_REWARD_WEIGHT,
      "velocity_scale": JOINT_VELOCITY_SMOOTHNESS_EXP_SCALE,
      "acceleration_scale": JOINT_ACCELERATION_SMOOTHNESS_EXP_SCALE,
      "action_rate_scale": ACTION_RATE_SMOOTHNESS_EXP_SCALE,
      "action_acceleration_scale": ACTION_ACCELERATION_SMOOTHNESS_EXP_SCALE,
      "action_magnitude_scale": ACTION_MAGNITUDE_SMOOTHNESS_EXP_SCALE,
      "joint_asset_cfg": SceneEntityCfg("robot", joint_names=(f"{hand}_.*",)),
      "joint_limit_penalty_weight": JOINT_POSITION_LIMITS_REWARD_WEIGHT,
      "palm_sensor_names": palm_sensor_names_by_side[hand],
      "palm_force_penalty_weight": PALM_GROUND_CONTACT_FORCE_PENALTY_WEIGHT,
      "action_term_name": "joint_pos",
    }

  def make_left_hand_internal_reward_params() -> dict[str, Any]:
    return {
      **make_single_hand_alignment_reward_params("left"),
      **make_single_hand_internal_common_params("left"),
    }

  def make_right_hand_internal_reward_params() -> dict[str, Any]:
    return {
      **make_single_hand_alignment_reward_params("right"),
      **make_single_hand_internal_common_params("right"),
    }

  rewards = {
    "right_hand_internal": RewardTermCfg(
      func=manipulation_mdp.right_hand_internal_reward,
      weight=1.0,
      params=make_right_hand_internal_reward_params(),
    ),
  }
  if not is_move_reward_mode:
    rewards["left_hand_internal"] = RewardTermCfg(
      func=manipulation_mdp.left_hand_internal_reward,
      weight=1.0,
      params=make_left_hand_internal_reward_params(),
    )

  if is_residual_reward_mode:
    rewards["right_contact_graph"] = RewardTermCfg(
      func=manipulation_mdp.single_hand_contact_graph_alignment_reward,
      weight=CONTACT_GRAPH_REWARD_WEIGHT,
      params=make_single_side_contact_graph_reward_params("right"),
    )
    rewards["right_key_fingertip_force"] = RewardTermCfg(
      func=manipulation_mdp.single_hand_key_fingertip_force_reward,
      weight=KEY_FINGERTIP_FORCE_REWARD_WEIGHT,
      params=make_single_side_key_fingertip_force_reward_params("right"),
    )
    if not is_move_reward_mode:
      rewards["left_contact_graph"] = RewardTermCfg(
        func=manipulation_mdp.single_hand_contact_graph_alignment_reward,
        weight=CONTACT_GRAPH_REWARD_WEIGHT,
        params=make_single_side_contact_graph_reward_params("left"),
      )
      rewards["left_key_fingertip_force"] = RewardTermCfg(
        func=manipulation_mdp.single_hand_key_fingertip_force_reward,
        weight=KEY_FINGERTIP_FORCE_REWARD_WEIGHT,
        params=make_single_side_key_fingertip_force_reward_params("left"),
      )
    rewards["key_fingertip_target_distance"] = RewardTermCfg(
      func=manipulation_mdp.key_fingertip_target_distance_penalty,
      weight=KEY_FINGERTIP_TARGET_DISTANCE_PENALTY_WEIGHT,
      params=make_key_fingertip_target_distance_penalty_params(),
    )
    contact_move_regularization_term_name = (
      "contact_move_right_speed"
      if is_move_reward_mode
      else "contact_move_object_regularization"
    )
    rewards[contact_move_regularization_term_name] = RewardTermCfg(
      func=manipulation_mdp.contact_move_object_regularization_reward,
      weight=CONTACT_MOVE_OBJECT_REGULARIZATION_REWARD_WEIGHT,
      params=make_contact_move_object_regularization_params(),
    )
  if is_move_reward_mode:
    rewards["right_object_reset_rotation_penalty"] = RewardTermCfg(
      func=manipulation_mdp.object_reset_rotation_penalty,
      weight=RIGHT_OBJECT_RESET_ROTATION_PENALTY_WEIGHT,
      params={
        "object_name": "right_obj",
        "rotation_threshold": RIGHT_OBJECT_RESET_ROTATION_PENALTY_THRESHOLD,
      },
    )
  if is_residual_reward_mode:
    rewards["contact_move_relative_pose"] = RewardTermCfg(
      func=manipulation_mdp.contact_move_relative_pose_reward,
      weight=CONTACT_MOVE_RELATIVE_POSE_REWARD_WEIGHT,
      params=make_contact_move_relative_pose_params(),
    )
  if is_move_reward_mode:
    rewards["right_object_stage"] = RewardTermCfg(
      func=manipulation_mdp.right_object_stage_reward,
      weight=RIGHT_OBJECT_STAGE_REWARD_WEIGHT,
      params=make_right_object_stage_params(),
    )

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }
  if manager_settings.invalid_physics_termination_params is not None:
    terminations["invalid_physics_state"] = TerminationTermCfg(
      func=manipulation_mdp.invalid_physics_state,
      params=dict(manager_settings.invalid_physics_termination_params),
    )
  if manager_settings.object_rotation_termination_params is not None:
    object_rotation_params = dict(manager_settings.object_rotation_termination_params)
    if is_residual_reward_mode:
      object_rotation_params["asset_names"] = ("left_obj",)
    terminations["object_rotation_limit"] = TerminationTermCfg(
      func=manipulation_mdp.object_rotation_over_limit,
      params=object_rotation_params,
    )
  if is_move_reward_mode:
    terminations["right_object_stage_success_count"] = TerminationTermCfg(
      func=manipulation_mdp.right_object_stage_success,
      params=make_right_object_stage_success_params(),
    )
    terminations["right_object_stage_no_progress"] = TerminationTermCfg(
      func=manipulation_mdp.right_object_stage_no_progress,
      params=make_right_object_stage_no_progress_params(),
    )
  if is_residual_reward_mode:
    terminations["key_fingertip_target_distance_failure"] = TerminationTermCfg(
      func=manipulation_mdp.key_fingertip_target_distance_over_limit,
      params=make_key_fingertip_target_distance_termination_params(),
    )

  curriculum = {}
  if is_move_reward_mode and not play:
    curriculum["right_object_target3"] = CurriculumTermCfg(
      func=manipulation_mdp.right_object_target3_curriculum,
      params=make_right_object_target3_curriculum_params(),
    )

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=1.0,
      sensors=scene_sensors,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=1.5,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=140,
      njmax=600,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=4,
    episode_length_s=TRAIN_EPISODE_LENGTH_S,
  )
  cfg.sim.mujoco.iterations = max(cfg.sim.mujoco.iterations, 20)
  cfg.sim.mujoco.ls_iterations = max(cfg.sim.mujoco.ls_iterations, 20)
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.mujoco.ccd_tolerance = 1e-3

  if play:
    cfg.episode_length_s = PLAY_EPISODE_LENGTH_S
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
  return cfg


def unitree_g1_wuji_lift_cube_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg(play=play)
  configure_lift_cube_scene(
    cfg,
    get_g1_wuji_manip_robot_cfg(),
    show_spawn_rect=play,
  )
  return cfg


def unitree_g1_wuji_contact_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg(play=play, reward_mode="contact")
  configure_lift_cube_scene(
    cfg,
    get_g1_wuji_manip_robot_cfg(),
    show_spawn_rect=play,
  )
  return cfg


def unitree_g1_wuji_move_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg(play=play, reward_mode="move")
  configure_lift_cube_scene(
    cfg,
    get_g1_wuji_manip_robot_cfg(),
    show_spawn_rect=play,
  )
  return cfg
