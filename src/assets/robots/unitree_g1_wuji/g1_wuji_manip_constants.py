"""Unitree G1 Wuji 上身 + 双手（`g1_wuji.xml`）manipulation；本文件自包含，不含腿/腰执行器设计。"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg, XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia_from_two_stage_planetary
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF
##

G1_WUJI_MANIP_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1_wuji" / "xmls" / "g1_wuji.xml"
)
assert G1_WUJI_MANIP_XML.exists()

HAND_COLLISION_PREFIXES = ("left_palm", "right_palm", "left_finger", "right_finger")
ARM_COLLISION_BODY_NAMES = {
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "left_wrist_roll_link",
  "left_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link",
}
TORSO_COLLISION_BODY_NAME = "torso_link"
TORSO_COLLISION_MESH_NAME = "torso_link"
FINGER_SITE_SUFFIXES = ("t", "i", "m", "r", "l")
HAND_JOINT_ALIGNMENT_BODIES = (
  ("joint1", "link1"),
  ("joint3", "link3"),
  ("joint4", "link4"),
)
HAND_JOINT_ALIGNMENT_SITE_SIZE = 0.004


def _iter_bodies(body: mujoco.MjsBody):
  yield body
  for child in body.bodies:
    yield from _iter_bodies(child)


def _configure_upper_body_collisions(spec: mujoco.MjSpec) -> None:
  """Preserve hand, arm, and torso collisions while disabling the rest."""
  for body in _iter_bodies(spec.worldbody):
    for geom in body.geoms:
      geom_name = geom.name or ""
      is_hand_collision = geom_name.endswith("_collision") and geom_name.startswith(
        HAND_COLLISION_PREFIXES
      )
      is_arm_collision = (
        body.name in ARM_COLLISION_BODY_NAMES
        and (int(geom.contype) != 0 or int(geom.conaffinity) != 0)
      )
      is_torso_collision = (
        body.name == TORSO_COLLISION_BODY_NAME
        and getattr(geom, "meshname", "") == TORSO_COLLISION_MESH_NAME
        and (int(geom.contype) != 0 or int(geom.conaffinity) != 0)
      )
      if is_hand_collision or is_arm_collision or is_torso_collision:
        continue
      geom.contype = 0
      geom.conaffinity = 0


def _ensure_hand_alignment_sites(spec: mujoco.MjSpec) -> None:
  """Add hand alignment sites at actual joint/body frames, not interpolated offsets."""
  existing_site_names = {site.name for site in spec.sites}
  for hand in ("left", "right"):
    for finger_id, suffix in enumerate(FINGER_SITE_SUFFIXES, start=1):
      fingertip_site = spec.site(f"{hand}_fingertip_site_{suffix}")
      rgba = tuple(float(channel) for channel in fingertip_site.rgba)

      for joint_name, link_name in HAND_JOINT_ALIGNMENT_BODIES:
        link = spec.body(f"{hand}_finger{finger_id}_{link_name}")
        site_name = f"{hand}_{joint_name}_site_{suffix}"
        if site_name not in existing_site_names:
          link.add_site(
            name=site_name,
            pos=(0.0, 0.0, 0.0),
            size=(HAND_JOINT_ALIGNMENT_SITE_SIZE,),
            rgba=rgba,
          )
          existing_site_names.add(site_name)


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, G1_WUJI_MANIP_XML.parent / "assets", meshdir)
  # 手部 mesh 以文件名形式在 g1_wuji.xml 中引用，这里显式注入左右手 STL。
  update_assets(assets, G1_WUJI_MANIP_XML.parent.parent / "meshes" / "left", meshdir)
  update_assets(assets, G1_WUJI_MANIP_XML.parent.parent / "meshes" / "right", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_WUJI_MANIP_XML))
  spec.assets = get_assets(spec.meshdir)
  _configure_upper_body_collisions(spec)
  _ensure_hand_alignment_sites(spec)
  return spec

## 5020（肩/肘/腕 roll）与 4010（腕 pitch/yaw）
ROTOR_INERTIAS_5020 = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_5020 = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

## 5020（肩/肘/腕 roll）与 4010（腕 pitch/yaw）
ROTOR_INERTIAS_4010 = (
  0.068e-4,
  0.0,
  0.0,
)
GEARS_4010 = (
  1,
  5,
  5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

## 5020（肩/肘/腕 roll）与 4010（腕 pitch/yaw）
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz → rad/s
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# 电机名义规格： 5020/4010
ACTUATOR_5020 = ElectricActuator(
  reflected_inertia=ARMATURE_5020,
  velocity_limit=37.0,
  effort_limit=25.0,
)
ACTUATOR_4010 = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=22.0,
  effort_limit=5.0,
)

G1_WUJI_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)

G1_WUJI_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
  effort_limit=ACTUATOR_4010.effort_limit,
  armature=ACTUATOR_4010.reflected_inertia,
)

## 手指：刚度、阻尼；
G1_WUJI_FINGER_JOINT_PD = (
  (".*_finger1_joint1", 0.40844710645048043, 0.020882010257063675, 0.4452),
  (".*_finger1_joint2", 0.6858601063643346, 0.030610939996373314, 0.4259),
  (".*_finger1_joint3", 0.2391112196099482, 0.010181475050560962, 0.1888),
  (".*_finger1_joint4", 0.20736128319711747, 0.00909698844675045, 0.1468),
  (".*_finger2_joint1", 0.37352218155860073, 0.01882274330029718, 0.6188),
  (".*_finger2_joint2", 0.45592448909027794, 0.019798167597016643, 0.1822),
  (".*_finger2_joint3", 0.24368366649522863, 0.010477031953162727, 0.2251),
  (".*_finger2_joint4", 0.18026971340925335, 0.008240212147903584, 0.217),
  (".*_finger3_joint1", 0.3687093483646485, 0.01848622487024593, 0.6494),
  (".*_finger3_joint2", 0.4164253443634641, 0.018032947229953678, 0.1827),
  (".*_finger3_joint3", 0.22218607502059182, 0.009592200014076666, 0.2078),
  (".*_finger3_joint4", 0.19427606072023446, 0.009152994605972402, 0.2018),
  (".*_finger4_joint1", 0.35718151495111794, 0.018376606800780005, 0.6389),
  (".*_finger4_joint2", 0.42977315313086895, 0.01867700966212433, 0.1832),
  (".*_finger4_joint3", 0.24930151196247122, 0.01059512121009555, 0.2249),
  (".*_finger4_joint4", 0.2285032688178066, 0.009917602877441107, 0.2044),
  (".*_finger5_joint1", 0.3655325975433942, 0.018616960278988272, 0.6441),
  (".*_finger5_joint2", 0.41393113081120425, 0.018732177029667153, 0.1798),
  (".*_finger5_joint3", 0.22729367621965954, 0.00951005441616486, 0.2384),
  (".*_finger5_joint4", 0.1964723550341816, 0.009017295241756363, 0.1866),
)

G1_WUJI_ACTUATOR_FINGERS = tuple(
  BuiltinPositionActuatorCfg(
    target_names_expr=(target_name,),
    stiffness=stiff,
    damping=damp,
    effort_limit=eff,
    armature=0.01,
  )
  for target_name, stiff, damp, eff in G1_WUJI_FINGER_JOINT_PD
)

# 机器人位置
G1_WUJI_MANIP_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "left_shoulder_roll_joint": 0.5,
    "right_shoulder_roll_joint": -0.5,
    "left_elbow_joint": -0.5,
    "right_elbow_joint": -0.5,
  },
  joint_vel={".*": 0.0},
)

# 碰撞（与全机描述一致；场景无脚时规则仍可保留）
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
  disable_other_geoms=False,
)

G1_WUJI_MANIP_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_WUJI_ACTUATOR_ARM,
    G1_WUJI_ACTUATOR_WRIST,
    *G1_WUJI_ACTUATOR_FINGERS,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_wuji_manip_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=G1_WUJI_MANIP_HOME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_WUJI_MANIP_ARTICULATION,
  )


G1_WUJI_MANIP_ACTION_SCALE: dict[str, float] = {}
for a in G1_WUJI_MANIP_ARTICULATION.actuators:
  if isinstance(a, BuiltinPositionActuatorCfg):
    e = a.effort_limit
    s = a.stiffness
    names = a.target_names_expr
    assert e is not None
    for n in names:
      if "_finger" in n:
        G1_WUJI_MANIP_ACTION_SCALE[n] = 0.25
      else:
        G1_WUJI_MANIP_ACTION_SCALE[n] = 0.25 * e / s
