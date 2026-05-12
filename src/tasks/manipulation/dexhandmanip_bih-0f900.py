from __future__ import annotations

import os
import pickle
import random
from itertools import cycle
from time import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from ...utils import torch_jit_utils as torch_jit_utils
from bps_torch.bps import bps_torch
from gym import spaces
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import normalize_angle, quat_conjugate, quat_mul
import math
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.factory import ManipDataFactory

from main.dataset.oakink2_dataset_dexhand_lh import OakInk2DatasetDexHandLH
from main.dataset.oakink2_dataset_dexhand_rh import OakInk2DatasetDexHandRH
from main.dataset.oakink2_dataset_utils import oakink2_obj_scale, oakink2_obj_mass
from main.dataset.transform import aa_to_quat, aa_to_rotmat, quat_to_rotmat, rotmat_to_aa, rotmat_to_quat
from torch import Tensor
from tqdm import tqdm
from ...asset_root import ASSET_ROOT


from ..core.config import ROBOT_HEIGHT, config
from ...envs.core.sim_config import sim_config
from ...envs.core.vec_task import VecTask
from ...utils.pose_utils import get_mat


class DexHandManipBiHEnv(VecTask):



    def __init__(
        self,
        cfg,
        *,
        rl_device: int = 0,
        sim_device: int = 0,
        graphics_device_id: int = 0,
        display: bool = False,
        record: bool = False,
        headless: bool = True,
    ):
        self._record = record
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.training = self.cfg["env"]["training"]
        
        self.dexhand_type = self.cfg["env"]["dexhand"]
        self.dexhand_rh = DexHandFactory.create_hand(self.dexhand_type, "right", headless=headless)
        self.dexhand_lh = DexHandFactory.create_hand(self.dexhand_type, "left", headless=headless)

        # 标志位：是否使用左右手（稍后在数据加载后会自动更新）
        self.use_rh = True
        self.use_lh = True
        
        # 暂时设置为双手，稍后会根据数据情况自动调整
        self.cfg["env"]["numActions"] = ((self.dexhand_lh.n_dofs)) * 2 
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        # a dict containing prop obs name to dump and their dimensions
        # used for distillation
        self._prop_dump_info = self.cfg["env"]["propDumpInfo"]

        # Values to be filled in at runtime
        self.rh_states = {}
        self.lh_states = {}
        self.dexhand_rh_handles = {}  # will be dict mapping names to relevant sim handles
        self.dexhand_lh_handles = {}  # will be dict mapping names to relevant sim handles
        self.objs_handles = {}  # for obj handlers
        self.objs_assets = {}
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed

        self.dataIndices = self.cfg["env"]["dataIndices"]
        self.obs_future_length = self.cfg["env"]["obsFutureLength"]
        self.rollout_state_init = self.cfg["env"]["rolloutStateInit"]
        self.random_state_init = self.cfg["env"]["randomStateInit"]

        # 使用指数衰减进行课程学习
        self.tighten_factor = self.cfg["env"]["tightenFactor"]
        self.tighten_steps = self.cfg["env"]["tightenSteps"]
        
        # 手动偏移参数（与retarget文件保持一致）
        self.manual_offset_dx = -0.0
        self.manual_offset_dy = 0.0
        
        # 训练模式配置
        self.center_grasp_mode = self.cfg["env"].get("center_grasp_mode", "none")
        
        # 成功率跟踪用于动态调整惩罚系数
        self.recent_success_rate = 0.0  # 初始成功率设为0.0
        
        print(f"ENV INIT: Center grasp mode = {self.center_grasp_mode}")
        
        # 可视化开关：默认关闭可视化代码（避免性能开销）
        self.enable_visualization = self.cfg["env"].get("enable_visualization", False)
        
        self.rollout_len = self.cfg["env"].get("rolloutLen", None)
        self.rollout_begin = self.cfg["env"].get("rolloutBegin", None)

        assert len(self.dataIndices) == 1 or self.rollout_len is None, "rolloutLen only works with one data"
        assert len(self.dataIndices) == 1 or self.rollout_begin is None, "rolloutBegin only works with one data"

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None  # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self.net_cf = None  # contact force
        self._eef_state = None  # end effector state (at grasping point)
        self._ftip_center_state = None  # center of fingertips
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._pos_control = None  # Position actions
        self._effort_control = None  # Torque actions
        self._dexhand_rh_effort_limits = None  # Actuator effort limits for dexhand_r
        self._dexhand_rh_dof_speed_limits = None  # Actuator speed limits for dexhand_r
        self._global_dexhand_rh_indices = None  # Unique indices corresponding to all envs in flattened array

        self.sim_device = torch.device(sim_device)
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )
        # 改观测 (删除了所有速度和手腕位置)
        TARGET_OBS_DIM = (
            128  # bps
            + 5  # gt_tips_distance
            + (
                (self.dexhand_rh.n_bodies - 1) * 3  # delta_joints_pos
                + 3  # delta_manip_obj_pos
                + 4  # manip_obj_quat
                + 4  # delta_manip_obj_quat
                + self.dexhand_rh.n_bodies  # obj_to_joints
            )
            * self.obs_future_length
        ) * 2
        self.obs_dict.update(
            {
                "target": torch.zeros((self.num_envs, TARGET_OBS_DIM), device=self.device),
            }
        )
        obs_space = self.obs_space.spaces
        obs_space["target"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(TARGET_OBS_DIM,),
        )
        self.obs_space = spaces.Dict(obs_space)

        # load BPS model
        self.bps_feat_type = "dists"
        self.bps_layer = bps_torch(
            bps_type="grid_sphere", n_bps_points=128, radius=0.2, randomize=False, device=self.device
        )

        obj_verts_rh = self.demo_data_rh["obj_verts"]
        self.obj_bps_rh = self.bps_layer.encode(obj_verts_rh, feature_type=self.bps_feat_type)[self.bps_feat_type]
        obj_verts_lh = self.demo_data_lh["obj_verts"]
        self.obj_bps_lh = self.bps_layer.encode(obj_verts_lh, feature_type=self.bps_feat_type)[self.bps_feat_type]

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.8 # 重力
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs()

        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # * >>> import table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True

        # 桌子和机器人位置
        table_asset = self.gym.create_box(self.sim, 0.9, 1.6, 0.03, table_asset_options)
        table_pos = gymapi.Vec3(0.1, 0, 0.85)
        self._table_surface_z = table_pos.z + 0.015
        self.robot_pose = gymapi.Transform()
        self.robot_pose.p = gymapi.Vec3(-0.65, 0.0, 0.794874)
        self.robot_pose.r = gymapi.Quat.from_euler_zyx(0, -np.pi / 2, 0)

        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        mujoco2gym_transf[:3, 3] = np.array([0, 0, self._table_surface_z])
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        dataset_list = list(set([ManipDataFactory.dataset_type(data_idx) for data_idx in self.dataIndices]))

        self.demo_dataset_lh_dict = {}
        self.demo_dataset_rh_dict = {}

        for dataset_type in dataset_list:
            self.demo_dataset_lh_dict[dataset_type] = ManipDataFactory.create_data(
                manipdata_type=dataset_type,
                side="left",
                device=self.sim_device,
                mujoco2gym_transf=self.mujoco2gym_transf,
                max_seq_len=self.max_episode_length,
                dexhand=self.dexhand_lh,
                embodiment=self.cfg["env"]["dexhand"],
            )
            self.demo_dataset_rh_dict[dataset_type] = ManipDataFactory.create_data(
                manipdata_type=dataset_type,
                side="right",
                device=self.sim_device,
                mujoco2gym_transf=self.mujoco2gym_transf,
                max_seq_len=self.max_episode_length,
                dexhand=self.dexhand_rh,
                embodiment=self.cfg["env"]["dexhand"],
            )

        # 使用完整机器人URDF（仅保留完整机器人路径）
        robot_asset_file = self.dexhand_rh.urdf_path
        asset_options = gymapi.AssetOptions()
        asset_options.thickness = 0.001
        asset_options.angular_damping = 20
        asset_options.linear_damping = 20
        asset_options.max_linear_velocity = 50
        asset_options.max_angular_velocity = 100
        asset_options.fix_base_link = True
        asset_options.disable_gravity = False
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False  # 不合并fixed关节，保持完整的机器人结构
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        full_robot_asset = self.gym.load_asset(self.sim, *os.path.split(robot_asset_file), asset_options)
        # 完整机器人的DOF配置（仅保留完整机器人路径）
        full_dof_count = self.gym.get_asset_dof_count(full_robot_asset)  # 例如 38 个 DOF
        controllable_dof_count = self.dexhand_rh.n_dofs + self.dexhand_lh.n_dofs
        print(f"Full Robot - Total DOFs: {full_dof_count}, Controllable DOFs: {controllable_dof_count}")
        full_dof_stiffness = torch.zeros(full_dof_count, dtype=torch.float, device=self.sim_device)
        full_dof_damping = torch.zeros(full_dof_count, dtype=torch.float, device=self.sim_device)
        # 刚度阻尼
        # 左侧臂腕手（左侧 19 DOF：7 臂 + 12 手指）
        full_dof_stiffness[0:3] = 150
        full_dof_stiffness[3] = 100
        full_dof_stiffness[4:7] = 40
        full_dof_stiffness[7:19] = 10

        full_dof_damping[0:3] = 5
        full_dof_damping[3] = 5
        full_dof_damping[4:7] = 4
        full_dof_damping[7:19] = 2
        # 右侧臂腕手（右侧 19 DOF：7 臂 + 12 手指）
        full_dof_stiffness[19:22] = 150
        full_dof_stiffness[22] = 100
        full_dof_stiffness[23:26] = 40
        full_dof_stiffness[26:38] = 10
        
        full_dof_damping[19:22] = 5
        full_dof_damping[22] = 5
        full_dof_damping[23:26] = 4
        full_dof_damping[26:38] = 2

        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(full_robot_asset)
        for element in rigid_shape_props_asset:
            element.friction = 4.0
            element.rolling_friction = 0.01
            element.torsion_friction = 0.01
        self.gym.set_asset_rigid_shape_properties(full_robot_asset, rigid_shape_props_asset)

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(full_robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(full_robot_asset)
        print(f"Num Robot Bodies: {self.num_robot_bodies}")
        print(f"Num Robot DOFs: {self.num_robot_dofs}")

        # 完整机器人的DOF属性设置
        full_dof_props = self.gym.get_asset_dof_properties(full_robot_asset)
        
        # 设置完整机器人的DOF属性
        for i in range(self.num_robot_dofs):  # 38个DOF
            full_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            full_dof_props["stiffness"][i] = full_dof_stiffness[i]
            full_dof_props["damping"][i] = full_dof_damping[i]
        
        # 提取左手DOF限制信息
        self.dexhand_lh_dof_lower_limits = []
        self.dexhand_lh_dof_upper_limits = []
        self._dexhand_lh_effort_limits = []
        self._dexhand_lh_dof_speed_limits = []
        for dof_idx in self.dexhand_lh.dof_indices:
            self.dexhand_lh_dof_lower_limits.append(full_dof_props["lower"][dof_idx])
            self.dexhand_lh_dof_upper_limits.append(full_dof_props["upper"][dof_idx])
            self._dexhand_lh_effort_limits.append(full_dof_props["effort"][dof_idx])
            self._dexhand_lh_dof_speed_limits.append(full_dof_props["velocity"][dof_idx])
        
        # 提取右手DOF限制信息
        self.dexhand_rh_dof_lower_limits = []
        self.dexhand_rh_dof_upper_limits = []
        self._dexhand_rh_effort_limits = []
        self._dexhand_rh_dof_speed_limits = []
        for dof_idx in self.dexhand_rh.dof_indices:
            self.dexhand_rh_dof_lower_limits.append(full_dof_props["lower"][dof_idx])
            self.dexhand_rh_dof_upper_limits.append(full_dof_props["upper"][dof_idx])
            self._dexhand_rh_effort_limits.append(full_dof_props["effort"][dof_idx])
            self._dexhand_rh_dof_speed_limits.append(full_dof_props["velocity"][dof_idx])
        
        # 转换为张量
        self.dexhand_lh_dof_lower_limits = torch.tensor(self.dexhand_lh_dof_lower_limits, device=self.sim_device)
        self.dexhand_lh_dof_upper_limits = torch.tensor(self.dexhand_lh_dof_upper_limits, device=self.sim_device)
        self._dexhand_lh_effort_limits = torch.tensor(self._dexhand_lh_effort_limits, device=self.sim_device)
        self._dexhand_lh_dof_speed_limits = torch.tensor(self._dexhand_lh_dof_speed_limits, device=self.sim_device)
        
        self.dexhand_rh_dof_lower_limits = torch.tensor(self.dexhand_rh_dof_lower_limits, device=self.sim_device)
        self.dexhand_rh_dof_upper_limits = torch.tensor(self.dexhand_rh_dof_upper_limits, device=self.sim_device)
        self._dexhand_rh_effort_limits = torch.tensor(self._dexhand_rh_effort_limits, device=self.sim_device)
        self._dexhand_rh_dof_speed_limits = torch.tensor(self._dexhand_rh_dof_speed_limits, device=self.sim_device)
    

        # 机器人刚体和形状数量
        num_robot_bodies = self.gym.get_asset_rigid_body_count(full_robot_asset)
        num_robot_shapes = self.gym.get_asset_rigid_shape_count(full_robot_asset)

        self.robot_actors = []
        self.envs = []

        assert len(self.dataIndices) == 1 or not self.rollout_state_init, "rollout_state_init only works with one data"

        def segment_data(k, data_dict, side):
            todo_list = self.dataIndices
            idx = todo_list[k % len(todo_list)]
            demo = data_dict[ManipDataFactory.dataset_type(idx)][idx]

            dex_name = self.dexhand_type
            pkl_path = f"data/retargeting/OakInk-v2/mano2{dex_name}_{'rh' if side == 'rh' else 'lh'}/{idx}.pkl"
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    saved = pickle.load(f)
                # 加载关节位置 [T, 19]
                if 'opt_dof_pos' in saved:
                    opt = torch.tensor(saved['opt_dof_pos'], device=self.sim_device, dtype=torch.float32)
                    demo['opt_dof_pos'] = opt
            return demo

        self.demo_data_lh = [segment_data(i, self.demo_dataset_lh_dict, side="lh") for i in tqdm(range(self.num_envs), desc="Loading LH data")]
        self.demo_data_lh = self.pack_data(self.demo_data_lh, side="lh")
        self.demo_data_rh = [segment_data(i, self.demo_dataset_rh_dict, side="rh") for i in tqdm(range(self.num_envs), desc="Loading RH data")]
        self.demo_data_rh = self.pack_data(self.demo_data_rh, side="rh")
        
        # 默认使用双手
        self.use_rh = True
        self.use_lh = True
        self.cfg["env"]["numActions"] = self.dexhand_lh.n_dofs * 2
        print(f"✓ 手部配置: 使用双手 (2只手, {self.cfg['env']['numActions']} DOFs)")

        # Create environments
        self.manip_obj_rh_mass = []
        self.manip_obj_rh_com = []
        self.manip_obj_lh_mass = []
        self.manip_obj_lh_com = []
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            rh_current_asset, rh_sum_rigid_body_count, rh_sum_rigid_shape_count, rh_obj_scale, rh_obj_mass = (
                self._create_obj_assets(i, side="rh")
            )
            lh_current_asset, lh_sum_rigid_body_count, lh_sum_rigid_shape_count, lh_obj_scale, lh_obj_mass = (
                self._create_obj_assets(i, side="lh")
            )

            max_agg_bodies = (
                num_robot_bodies # 机器人刚体数量（已包含左右手）
                + 1 # 桌子
                + rh_sum_rigid_body_count # 右手物体刚体数量
                + lh_sum_rigid_body_count # 左手物体刚体数量
            )  # 1 for table
            max_agg_shapes = (
                num_robot_shapes # 机器人形状数量（已包含左右手）
                + 1 # 桌子
                + rh_sum_rigid_shape_count # 右手物体形状数量
                + lh_sum_rigid_shape_count # 左手物体形状数量
                + (1 if self._record else 0) # 录制时的额外形状
            )
            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: dexhand_r should ALWAYS be loaded first in sim!
            # Always enable aggregation for performance, regardless of viewer mode
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # camera handler for view rendering
            if self.camera_handlers is not None:
                self.camera_handlers.append(
                    self.create_camera(
                        env=env_ptr,
                        isaac_gym=self.gym,
                    )
                )

            self.robot_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0.0)
            # 创建完整机器人
            full_robot_actor = self.gym.create_actor(
                env_ptr,
                full_robot_asset,  # 使用完整机器人资产
                self.robot_pose,  # 使用统一的机器人pose
                "full_robot",
                i,
                1,  # 启用自碰撞
            )
            
            # 启用DOF力传感器
            self.gym.enable_actor_dof_force_sensors(env_ptr, full_robot_actor)
            self.gym.set_actor_dof_properties(env_ptr, full_robot_actor, full_dof_props)

            # Create table and obstacles
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(table_pos.x, table_pos.y, table_pos.z)
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0)
            table_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_actor)
            table_props[0].friction = 0.1  # ? only one table shape in each env
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, table_props)
            # set table's color to be dark gray
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))

            # 创建物体actor
            self.obj_rh_handle, _ = self._create_obj_actor(env_ptr, i, rh_current_asset, side="rh")  
            self.obj_lh_handle, _ = self._create_obj_actor(env_ptr, i, lh_current_asset, side="lh")
            # 设置物体scale
            self.gym.set_actor_scale(env_ptr, self.obj_rh_handle, rh_obj_scale)
            self.gym.set_actor_scale(env_ptr, self.obj_lh_handle, lh_obj_scale)
            # 获取物体属性
            obj_props_rh = self.gym.get_actor_rigid_body_properties(env_ptr, self.obj_rh_handle)
            obj_props_lh = self.gym.get_actor_rigid_body_properties(env_ptr, self.obj_lh_handle)
            obj_props_rh[0].mass = min(0.5, obj_props_rh[0].mass)  
            obj_props_lh[0].mass = min(0.5, obj_props_lh[0].mass) 

            # 应用指定质量（若存在）
            if rh_obj_mass is not None:
                obj_props_rh[0].mass = rh_obj_mass
            if lh_obj_mass is not None:
                obj_props_lh[0].mass = lh_obj_mass

            # 应用物体属性
            self.gym.set_actor_rigid_body_properties(env_ptr, self.obj_rh_handle, obj_props_rh)
            self.gym.set_actor_rigid_body_properties(env_ptr, self.obj_lh_handle, obj_props_lh)
            # 存储物体质量 和 质心
            self.manip_obj_rh_mass.append(obj_props_rh[0].mass)
            self.manip_obj_rh_com.append(torch.tensor([obj_props_rh[0].com.x, obj_props_rh[0].com.y, obj_props_rh[0].com.z]))
            self.manip_obj_lh_mass.append(obj_props_lh[0].mass)
            self.manip_obj_lh_com.append(torch.tensor([obj_props_lh[0].com.x, obj_props_lh[0].com.y, obj_props_lh[0].com.z]))
            # 结束聚合
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.robot_actors.append(full_robot_actor)

        self.manip_obj_rh_mass = torch.tensor(self.manip_obj_rh_mass, device=self.device)
        self.manip_obj_rh_com = torch.stack(self.manip_obj_rh_com, dim=0).to(self.device)
        self.manip_obj_lh_mass = torch.tensor(self.manip_obj_lh_mass, device=self.device)
        self.manip_obj_lh_com = torch.stack(self.manip_obj_lh_com, dim=0).to(self.device)

        # 内存分配和对齐 内存对齐
        self.gym.prepare_sim(self.sim)
        
        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        # Resolve actor handles based on whether we use a single full-robot actor or two separate hand actors
        full_robot_handle = self.gym.find_actor_handle(env_ptr, "full_robot")
        dexhand_rh_handle = full_robot_handle
        dexhand_lh_handle = full_robot_handle
        self.dexhand_rh_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_rh_handle, k) for k in self.dexhand_rh.body_names
        }
        self.dexhand_lh_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_lh_handle, k) for k in self.dexhand_lh.body_names
        }
        
        self.dexhand_rh_cf_weights = {
            k: (1.0 if ("index_2" in k or "middle_2" in k or "ring_2" in k or "little_2" in k or "thumb_3" in k or "thumb_4" in k) else 0.0) for k in self.dexhand_rh.body_names
        }
        self.dexhand_lh_cf_weights = {
            k: (1.0 if ("index_2" in k or "middle_2" in k or "ring_2" in k or "little_2" in k or "thumb_3" in k or "thumb_4" in k) else 0.0) for k in self.dexhand_lh.body_names
        }







        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        
        # 打印 Isaac Gym 实际得到的关节顺序
        print("=== Isaac Gym 实际关节顺序 ===")
        for env_id in range(self.num_envs):
            env_ptr = self.envs[env_id]
            dof_names = self.gym.get_actor_dof_names(env_ptr, 0)  # 第一个actor (full_robot)
            print(f"环境 {env_id} 的关节顺序:")
            for i, dof_name in enumerate(dof_names):
                print(f"  DOF索引 {i}: {dof_name}")
            break  # 只打印第一个环境的关节顺序

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        _dof_force = self.gym.acquire_dof_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]

        self._manip_obj_rh_handle = self.gym.find_actor_handle(env_ptr, "manip_obj_rh")
        self._manip_obj_rh_root_state = self._root_state[:, self._manip_obj_rh_handle, :]
        self._manip_obj_lh_handle = self.gym.find_actor_handle(env_ptr, "manip_obj_lh")
        self._manip_obj_lh_root_state = self._root_state[:, self._manip_obj_lh_handle, :]

        self.net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        self.dof_force = gymtorch.wrap_tensor(_dof_force).view(self.num_envs, -1)
        self._manip_obj_rh_rigid_body_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, self._manip_obj_rh_handle, "base"
        )
        self._manip_obj_lh_rigid_body_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, self._manip_obj_lh_handle, "base"
        )
        self._manip_obj_rh_cf = self.net_cf[:, self._manip_obj_rh_rigid_body_handle, :]
        self._manip_obj_lh_cf = self.net_cf[:, self._manip_obj_lh_rigid_body_handle, :]

        self.dexhand_rh_root_state = self._root_state[:, dexhand_rh_handle, :]
        self.dexhand_lh_root_state = self._root_state[:, dexhand_lh_handle, :]

        self.apply_forces = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.apply_torque = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self._prev_dof_vel = torch.zeros((self.num_envs, self.dexhand_rh.n_dofs), dtype=torch.float, device=self.device)
        self._prev_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize indices
        dexhand_actor_name_rh = "full_robot"
        dexhand_actor_name_lh = "full_robot"

        self._global_dexhand_rh_indices = torch.tensor(
            [self.gym.find_actor_index(env, dexhand_actor_name_rh, gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)
        self._global_dexhand_lh_indices = torch.tensor(
            [self.gym.find_actor_index(env, dexhand_actor_name_lh, gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)

        self._global_manip_obj_rh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "manip_obj_rh", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)
        self._global_manip_obj_lh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "manip_obj_lh", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)

        CONTACT_HISTORY_LEN = 5
        self.rh_tips_contact_history = torch.ones(self.num_envs, CONTACT_HISTORY_LEN, 5, device=self.device).bool()
        self.lh_tips_contact_history = torch.ones(self.num_envs, CONTACT_HISTORY_LEN, 5, device=self.device).bool()

    def pack_data(self, data, side="rh"):
        packed_data = {}
        packed_data["seq_len"] = torch.tensor([len(d["obj_trajectory"]) for d in data], device=self.device)
        max_len = packed_data["seq_len"].max()
        assert max_len <= self.max_episode_length, "max_len should be less than max_episode_length"

        def fill_data(stack_data):
            for i in range(len(stack_data)):
                if len(stack_data[i]) < max_len:
                    stack_data[i] = torch.cat(
                        [
                            stack_data[i],
                            stack_data[i][-1]
                            .unsqueeze(0)
                            .repeat(max_len - len(stack_data[i]), *[1 for _ in stack_data[i].shape[1:]]),
                        ],
                        dim=0,
                    )
            return torch.stack(stack_data).squeeze()

        for k in data[0].keys():
            if k == "mano_joints" or k == "mano_joints_velocity":
                mano_joints = []
                for d in data:
                    if side == "rh":
                        mano_joints.append(
                            torch.concat(
                                [
                                    d[k][self.dexhand_rh.to_hand(j_name)[0]]
                                    for j_name in self.dexhand_rh.body_names
                                    if self.dexhand_rh.to_hand(j_name)[0] != "wrist"
                                ],
                                dim=-1,
                            )
                        )
                    else:
                        mano_joints.append(
                            torch.concat(
                                [
                                    d[k][self.dexhand_lh.to_hand(j_name)[0]]
                                    for j_name in self.dexhand_lh.body_names
                                    if self.dexhand_lh.to_hand(j_name)[0] != "wrist"
                                ],
                                dim=-1,
                            )
                        )
                packed_data[k] = fill_data(mano_joints)
            elif type(data[0][k]) == torch.Tensor:
                stack_data = [d[k] for d in data]
                if k != "obj_verts":
                    packed_data[k] = fill_data(stack_data)
                else:
                    packed_data[k] = torch.stack(stack_data).squeeze()
            elif type(data[0][k]) == np.ndarray:
                raise RuntimeError("Using np is very slow.")
            else:
                packed_data[k] = [d[k] for d in data]
        return packed_data

    def allocate_buffers(self):
        # will also allocate extra buffers for data dumping, used for distillation
        super().allocate_buffers()

        # basic prop fields
        if not self.training:
            self.dump_fileds = {
                k: torch.zeros(
                    (self.num_envs, v),
                    device=self.device,
                    dtype=torch.float,
                )
                for k, v in self._prop_dump_info.items()
            }

    def _create_obj_assets(self, i, side="rh"): # 物体的属性模板
        if side == "rh":
            obj_id = self.demo_data_rh["obj_id"][i]
        else:
            obj_id = self.demo_data_lh["obj_id"][i]

        if obj_id in self.objs_assets:
            current_asset = self.objs_assets[obj_id]
        else:
            asset_options = gymapi.AssetOptions()
            asset_options.override_com = True
            asset_options.override_inertia = True
            asset_options.convex_decomposition_from_submeshes = True
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.thickness = 0.001
            asset_options.max_linear_velocity = 50
            asset_options.max_angular_velocity = 100
            asset_options.fix_base_link = False
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 200000  # 降低凸分解分辨率以减少内存占用 (原值: 200000)
            asset_options.vhacd_params.max_convex_hulls = 32  # 限制最大凸包数量
            asset_options.density = 200  # * the average density of low-fill-rate 3D-printed models
            if side == "rh":
                obj_urdf_path = self.demo_data_rh["obj_urdf_path"][i]
            else:
                obj_urdf_path = self.demo_data_lh["obj_urdf_path"][i]
            current_asset = self.gym.load_asset(self.sim, *os.path.split(obj_urdf_path), asset_options)

            rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(current_asset)
            for element in rigid_shape_props_asset:
                element.friction = 2.0  # 物体摩擦力
                # * We increase the friction coefficient to compensate for missing skin deformation friction in simulation. See the Appx for details.
                element.rolling_friction = 0.05
                element.torsion_friction = 0.05
            self.gym.set_asset_rigid_shape_properties(current_asset, rigid_shape_props_asset)
            self.objs_assets[obj_id] = current_asset

        # * load assigned scale and mass for the object if available
        if obj_id in oakink2_obj_scale:
            scale = oakink2_obj_scale[obj_id]
        else:
            scale = 1.0

        if obj_id in oakink2_obj_mass:
            mass = oakink2_obj_mass[obj_id]
        else:
            mass = None

        sum_rigid_body_count = self.gym.get_asset_rigid_body_count(current_asset)
        sum_rigid_shape_count = self.gym.get_asset_rigid_shape_count(current_asset)
        return current_asset, sum_rigid_body_count, sum_rigid_shape_count, scale, mass

    def _create_obj_actor(self, env_ptr, i, current_asset, side="rh"): # 物体的实例化

        if side == "rh":
            obj_transf = self.demo_data_rh["obj_trajectory"][i][0]
        else:
            obj_transf = self.demo_data_lh["obj_trajectory"][i][0]

        pose = gymapi.Transform()
        # 应用手动偏移到物体初始位置
        obj_x = obj_transf[0, 3] + self.manual_offset_dx
        obj_y = obj_transf[1, 3] + self.manual_offset_dy
        obj_z = obj_transf[2, 3]
        pose.p = gymapi.Vec3(obj_x, obj_y, obj_z)
        obj_aa = rotmat_to_aa(obj_transf[:3, :3])
        obj_aa_angle = torch.norm(obj_aa)
        obj_aa_axis = obj_aa / obj_aa_angle
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(obj_aa_axis[0], obj_aa_axis[1], obj_aa_axis[2]), obj_aa_angle)

        # ? object actor filter bit is always 1
        if side == "rh":
            obj_actor = self.gym.create_actor(env_ptr, current_asset, pose, "manip_obj_rh", i, 0)
        else:
            obj_actor = self.gym.create_actor(env_ptr, current_asset, pose, "manip_obj_lh", i, 0)
        obj_index = self.gym.get_actor_index(env_ptr, obj_actor, gymapi.DOMAIN_SIM)

        if side == "rh":
            scene_objs = self.demo_data_rh["scene_objs"][i]
        else:
            scene_objs = self.demo_data_lh["scene_objs"][i]
        scene_asset_options = gymapi.AssetOptions()
        scene_asset_options.fix_base_link = True

        for so_id, scene_obj in enumerate(scene_objs):
            scene_obj_type = scene_obj["obj"].type
            scene_obj_size = scene_obj["obj"].size
            scene_obj_pose = scene_obj["pose"]
            if scene_obj_type == "cube":
                scene_asset = self.gym.create_box(
                    self.sim,
                    scene_obj_size[0],
                    scene_obj_size[1],
                    scene_obj_size[2],
                    scene_asset_options,
                )
                offset = np.eye(4)
                offset[:3, 3] = np.array(scene_obj_size) / 2
                scene_obj_pose = scene_obj_pose @ offset
            elif scene_obj_type == "cylinder":
                scene_asset = self.gym.create_box(
                    self.sim,
                    scene_obj_size[0] * 2,
                    scene_obj_size[0] * 2,
                    scene_obj_size[1],
                    scene_asset_options,
                )
            else:
                raise NotImplementedError
            scene_obj_pose = self.mujoco2gym_transf @ torch.tensor(
                scene_obj_pose, device=self.sim_device, dtype=torch.float32
            )
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(scene_obj_pose[0, 3], scene_obj_pose[1, 3], scene_obj_pose[2, 3])
            obj_aa = rotmat_to_aa(scene_obj_pose[:3, :3])
            obj_aa_angle = torch.norm(obj_aa)
            obj_aa_axis = obj_aa / obj_aa_angle
            pose.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(obj_aa_axis[0], obj_aa_axis[1], obj_aa_axis[2]), obj_aa_angle
            )
            self.gym.create_actor(env_ptr, scene_asset, pose, f"scene_obj_{so_id}", i, 0)
        # add dummy scene object
        MAX_SCENE_OBJS = 0 
        for so_id in range(MAX_SCENE_OBJS - len(scene_objs)):
            scene_asset = self.gym.create_box(self.sim, 0.02, 0.04, 0.06, scene_asset_options)
            # ? collision filter bit is always 0b11111111, never collide with anything (except the ground)
            a = self.gym.create_actor(
                env_ptr,
                scene_asset,
                gymapi.Transform(),
                f"scene_obj_{so_id +  len(scene_objs)}",
                self.num_envs + 1,
                0b1,
            )
            c = [
                gymapi.Vec3(1, 1, 0.5),
                gymapi.Vec3(0.5, 1, 1),
                gymapi.Vec3(1, 0, 1),
                gymapi.Vec3(1, 1, 0),
                gymapi.Vec3(0, 1, 1),
                gymapi.Vec3(0, 0, 1),
                gymapi.Vec3(0, 1, 0),
                gymapi.Vec3(1, 0, 0),
            ][so_id + len(scene_objs)]
            self.gym.set_rigid_body_color(env_ptr, a, 0, gymapi.MESH_VISUAL, c)

        # * just for visualization purposes, add a small sphere at the finger positions (optional)
        if not self.headless and self.enable_visualization:
            dexhand_template = self.dexhand_rh if side == "rh" else self.dexhand_lh
            for joint_vis_id, joint_name in enumerate(dexhand_template.body_names):
                joint_name = dexhand_template.to_hand(joint_name)[0]
                joint_point = self.gym.create_sphere(self.sim, 0.005, scene_asset_options)
                a = self.gym.create_actor(
                    env_ptr,
                    joint_point,
                    gymapi.Transform(),
                    f"{side}_mano_joint_{joint_vis_id}",
                    self.num_envs + 1,
                    0b1,
                )
                if "index" in joint_name:
                    inter_c = 70
                elif "middle" in joint_name:
                    inter_c = 130
                elif "ring" in joint_name:
                    inter_c = 190
                elif "pinky" in joint_name:
                    inter_c = 250
                elif "thumb" in joint_name:
                    inter_c = 10
                else:
                    inter_c = 0
                if "thumb_force_sensor_4" or "index_force_sensor_3" or "middle_force_sensor_3" or "ring_force_sensor_3" or "little_force_sensor_3" in joint_name:
                    c = gymapi.Vec3(inter_c / 255, 200 / 255, 200 / 255)
                elif "index_1" or "middle_1" or "ring_1" or "little_1" or "thumb_1" or "thumb_2" in joint_name:
                    c = gymapi.Vec3(200 / 255, inter_c / 255, 200 / 255)
                elif "index_2" or "middle_2" or "ring_2" or "little_2" or "thumb_3" in joint_name:
                    c = gymapi.Vec3(200 / 255, 200 / 255, inter_c / 255)
                else:
                    c = gymapi.Vec3(100 / 255, 150 / 255, 200 / 255)
                self.gym.set_rigid_body_color(env_ptr, a, 0, gymapi.MESH_VISUAL, c)









        return obj_actor, obj_index

    def _update_states(self):
        # 完整机器人模式下，右手DOF映射到19-37
        rh_dof_indices = self.dexhand_rh.dof_indices
        self.rh_states.update(
            {
                "q": self._q[:, rh_dof_indices],
                "cos_q": torch.cos(self._q[:, rh_dof_indices]),
                "sin_q": torch.sin(self._q[:, rh_dof_indices]),
                "dq": self._qd[:, rh_dof_indices],
            }
        )

        self.rh_states["joints_state"] = torch.stack(
            [self._rigid_body_state[:, self.dexhand_rh_handles[k], :] for k in self.dexhand_rh.body_names],
            dim=1,
        )
        self.rh_states.update(
            {
                "manip_obj_pos": self._manip_obj_rh_root_state[:, :3],
                "manip_obj_quat": self._manip_obj_rh_root_state[:, 3:7],
                "manip_obj_vel": self._manip_obj_rh_root_state[:, 7:10],
                "manip_obj_ang_vel": self._manip_obj_rh_root_state[:, 10:],
            }
        )

        # 完整机器人模式下，左手DOF映射到0-18
        lh_dof_indices = self.dexhand_lh.dof_indices
        self.lh_states.update(
            {
                "q": self._q[:, lh_dof_indices],
                "cos_q": torch.cos(self._q[:, lh_dof_indices]),
                "sin_q": torch.sin(self._q[:, lh_dof_indices]),
                "dq": self._qd[:, lh_dof_indices],
            }
        )
        self.lh_states["joints_state"] = torch.stack(
            [self._rigid_body_state[:, self.dexhand_lh_handles[k], :] for k in self.dexhand_lh.body_names],
            dim=1,
        )
        self.lh_states.update(
            {
                "manip_obj_pos": self._manip_obj_lh_root_state[:, :3],
                "manip_obj_quat": self._manip_obj_lh_root_state[:, 3:7],
                "manip_obj_vel": self._manip_obj_lh_root_state[:, 7:10],
                "manip_obj_ang_vel": self._manip_obj_lh_root_state[:, 10:],
            }
        )

    def _refresh(self):

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        # 根据手部配置计算奖励
        reward_bufs = []
        reset_bufs = []
        success_bufs = []
        failure_bufs = []
        reward_dicts = {}
        error_bufs = []
        
        if self.use_lh:
            lh_rew_buf, lh_reset_buf, lh_success_buf, lh_failure_buf, lh_reward_dict, lh_error_buf = (
                self.compute_reward_side(actions, side="lh")
            )
            reward_bufs.append(lh_rew_buf)
            reset_bufs.append(lh_reset_buf)
            success_bufs.append(lh_success_buf)
            failure_bufs.append(lh_failure_buf)
            error_bufs.append(lh_error_buf)
            reward_dicts.update({"lh_" + k: v for k, v in lh_reward_dict.items()})
        
        if self.use_rh:
            rh_rew_buf, rh_reset_buf, rh_success_buf, rh_failure_buf, rh_reward_dict, rh_error_buf = (
                self.compute_reward_side(actions, side="rh")
            )
            reward_bufs.append(rh_rew_buf)
            reset_bufs.append(rh_reset_buf)
            success_bufs.append(rh_success_buf)
            failure_bufs.append(rh_failure_buf)
            error_bufs.append(rh_error_buf)
            reward_dicts.update({"rh_" + k: v for k, v in rh_reward_dict.items()})
        
        # 组合奖励和状态
        self.rew_buf = sum(reward_bufs)
        self.reset_buf = torch.stack(reset_bufs).any(dim=0) if len(reset_bufs) > 1 else reset_bufs[0]
        self.success_buf = torch.stack(success_bufs).all(dim=0) if len(success_bufs) > 1 else success_bufs[0]
        self.failure_buf = torch.stack(failure_bufs).any(dim=0) if len(failure_bufs) > 1 else failure_bufs[0]
        self.error_buf = torch.stack(error_bufs).any(dim=0) if len(error_bufs) > 1 else error_bufs[0]
        self.reward_dict = reward_dicts
        

    def compute_reward_side(self, actions, side="rh"):
        side_demo_data = self.demo_data_rh if side == "rh" else self.demo_data_lh
        target_state = {}
        max_length = torch.clip(side_demo_data["seq_len"], 0, self.max_episode_length).float()
        if self.rollout_len is not None:
            max_length = torch.clamp(max_length, 0, self.rollout_begin + self.rollout_len)

        cur_idx = self.progress_buf
        # 限制索引在有效范围内，防止越界
        max_idx = side_demo_data["wrist_pos"].shape[1] - 1
        cur_idx = torch.clamp(cur_idx, min=0, max=max_idx)
        cur_wrist_pos = side_demo_data["wrist_pos"][torch.arange(self.num_envs), cur_idx]
        # 应用手动偏移
        if torch.isnan(cur_wrist_pos).any() or torch.isinf(cur_wrist_pos).any():
            print("Warning: cur_wrist_pos contains NaN or inf values before offset")
            cur_wrist_pos = torch.zeros_like(cur_wrist_pos)
        cur_wrist_pos[:, 0] = cur_wrist_pos[:, 0] + self.manual_offset_dx
        cur_wrist_pos[:, 1] = cur_wrist_pos[:, 1] + self.manual_offset_dy
        target_state["wrist_pos"] = cur_wrist_pos

        target_state["tips_distance"] = side_demo_data["tips_distance"][torch.arange(self.num_envs), cur_idx]

        cur_joints_pos = side_demo_data["mano_joints"][torch.arange(self.num_envs), cur_idx]
        # 应用手动偏移
        if torch.isnan(cur_joints_pos).any() or torch.isinf(cur_joints_pos).any():
            print("Warning: cur_joints_pos contains NaN or inf values before offset")
            cur_joints_pos = torch.zeros_like(cur_joints_pos)
        cur_joints_pos[:, 0::3] = cur_joints_pos[:, 0::3] + self.manual_offset_dx
        cur_joints_pos[:, 1::3] = cur_joints_pos[:, 1::3] + self.manual_offset_dy
        target_state["joints_pos"] = cur_joints_pos.reshape(self.num_envs, -1, 3)

        cur_obj_transf = side_demo_data["obj_trajectory"][torch.arange(self.num_envs), cur_idx]
        # 应用手动偏移到物体位置
        obj_pos = cur_obj_transf[:, :3, 3].clone()
        obj_pos[:, 0] = obj_pos[:, 0] + self.manual_offset_dx
        obj_pos[:, 1] = obj_pos[:, 1] + self.manual_offset_dy
        target_state["manip_obj_pos"] = obj_pos
        target_state["manip_obj_quat"] = rotmat_to_quat(cur_obj_transf[:, :3, :3])[:, [1, 2, 3, 0]]
        
        # 计算参考的相对位置 (指尖相对于物体的位置)
        # 这对于交互自然性至关重要
        target_state["relative_pos"] = target_state["joints_pos"] - target_state["manip_obj_pos"][:, None, :]  # [N, n_joints, 3]

        target_state["manip_obj_vel"] = side_demo_data["obj_velocity"][torch.arange(self.num_envs), cur_idx]
        target_state["manip_obj_ang_vel"] = side_demo_data["obj_angular_velocity"][torch.arange(self.num_envs), cur_idx]
        
        # 参考速度
        dof_vel = side_demo_data["opt_dof_velocity"][torch.arange(self.num_envs), cur_idx]
        dof_vel = torch_jit_utils.tensor_clamp(
            dof_vel,
            -1 * getattr(self, f"_dexhand_{side}_dof_speed_limits").unsqueeze(0),
            getattr(self, f"_dexhand_{side}_dof_speed_limits").unsqueeze(0),
        )
        target_state["dof_vel"] = dof_vel

        target_state["tip_force"] = torch.stack(
            [
                self.net_cf[:, getattr(self, f"dexhand_{side}_handles")[k], :]
                for k in (self.dexhand_rh.contact_body_names if side == "rh" else self.dexhand_lh.contact_body_names)
            ],
            axis=1,
        )
        
        setattr(
            self,
            f"{side}_tips_contact_history",
            torch.concat(
                [
                    getattr(self, f"{side}_tips_contact_history")[:, 1:],
                    (torch.norm(target_state["tip_force"], dim=-1) > 0)[:, None],
                ],
                dim=1,
            ),
        )
        target_state["tip_contact_state"] = getattr(self, f"{side}_tips_contact_history")

        # ========== 接触图 ==========
        # 1. 计算当前仿真的Contact Graph状态 (5个指尖与物体的接触)
        fingertip_distances = target_state["tips_distance"]  # [N, 5]
        fingertip_forces = target_state["tip_force"]  # [N, 5, 3]
        finger_force_magnitudes = torch.norm(fingertip_forces, dim=-1)  # [N, 5]
        
        # 双重条件判断接触:
        # 条件1: 距离 < 2cm (几何接近)
        # 条件2: 接触力 > 0 (有真实接触力)
        contact_distance_threshold = 0.02  # 2cm
        contact_force_threshold = 0.0     # 只要有力就算接触
        
        distance_contact = fingertip_distances < contact_distance_threshold  # [N, 5]
        force_contact = finger_force_magnitudes > contact_force_threshold    # [N, 5]
        
        # 两个条件都满足才算接触
        contact_graph_current = (distance_contact & force_contact).float()  # [N, 5]
        target_state["contact_graph_current"] = contact_graph_current
        
        # 2. 获取参考的Contact Graph状态
        contact_graph_target = side_demo_data["tips_contact_graph"][torch.arange(self.num_envs), cur_idx]  # [N, 5]
        target_state["contact_graph_target"] = contact_graph_target

        side_states = getattr(self, f"{side}_states")
        if side == "rh":
            # 完整机器人模式下，右手DOF映射到19-37
            rh_dof_forces = self.dof_force[:, self.dexhand_rh.dof_indices]
            power = torch.abs(torch.multiply(rh_dof_forces, side_states["dq"])).sum(dim=-1)
        else:
            # 完整机器人模式下，左手DOF映射到0-18
            lh_dof_forces = self.dof_force[:, self.dexhand_lh.dof_indices]
            power = torch.abs(torch.multiply(lh_dof_forces, side_states["dq"])).sum(dim=-1)
        target_state["power"] = power

        if self.training:
            # 使用指数衰减进行课程学习（从易到难）
            last_step = self.gym.get_frame_count(self.sim)
            scale_factor = (np.e * 2) ** (-1 * last_step / self.tighten_steps) * (1 - self.tighten_factor) + self.tighten_factor
        else:
            # 测试模式使用最严格阈值，但裁剪到最小 0.75
            scale_factor = max(self.tighten_factor, 0.7)

        self.scale_factor = scale_factor

       # 计算关节加速度
        current_controllable_dq = side_states["dq"]
        dof_acc = (
            (current_controllable_dq - self._prev_dof_vel) / self.dt
        )
        # 计算action rate（动作变化率）的L2惩罚输入
        current_actions = self.actions if self.actions is not None else torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)
        action_rate = current_actions - self._prev_actions

        rew_buf, reset_buf, success_buf, failure_buf, reward_dict = compute_imitation_reward(
            self.reset_buf,
            self.progress_buf,
            self.running_progress_buf,
            actions,
            side_states,
            target_state,
            max_length.int().tolist(),
            scale_factor,
            self.recent_success_rate,  # 传递成功率参数
            (self.dexhand_rh if side == "rh" else self.dexhand_lh).weight_idx,
            side,  # 传递side参数
            self._rigid_body_state,
            dof_acc,
            action_rate,
        )

        # 更新上一时刻速度缓存
        self._prev_dof_vel = current_controllable_dq.clone()
        self._prev_actions = current_actions.clone()

        self.total_rew_buf += rew_buf
        return rew_buf, reset_buf, success_buf, failure_buf, reward_dict, torch.zeros_like(reset_buf)

    def compute_observations(self):
        self._refresh()
        
        # 根据手部配置收集观察
        obs_list = []
        if self.use_rh:
            obs_list.append(self.compute_observations_side("rh"))
        if self.use_lh:
            obs_list.append(self.compute_observations_side("lh"))
        
        # 组合观察
        if len(obs_list) == 1:
            self.obs_dict = obs_list[0]
        else:
            # 双手情况：拼接观察
            for k in obs_list[0].keys():
                self.obs_dict[k] = torch.cat([obs[k] for obs in obs_list], dim=-1)

    def compute_observations_side(self, side="rh"):
        side_states = getattr(self, f"{side}_states")
        side_demo_data = getattr(self, f"demo_data_{side}")

        obs_dict = {}

        obs_values = []
        for ob in self._obs_keys: # ResDexHand.yaml - obsKeys
            obs_values.append(side_states[ob]) # 状态观测
        obs_dict["proprioception"] = torch.cat(obs_values, dim=-1) # 拼接所有状态观测 - 本体感知观测

        # ResDexHand.yaml - privileged_obs_keys: dq, manip_obj_pos, manip_obj_quat, tip_force
        if len(self._privileged_obs_keys) > 0:
            pri_obs_values = []
            for ob in self._privileged_obs_keys:
                if ob == "manip_obj_pos": # 物体位置观测
                    wrist_pos = side_states["joints_state"][:, 0, :3]
                    pri_obs_values.append(side_states[ob] - wrist_pos)
                elif ob == "tip_force": # 指尖接触力观测 - 取预定义的接触刚体力信息
                    tip_force = torch.stack(
                        [
                            self.net_cf[:, getattr(self, f"dexhand_{side}_handles")[k], :]
                            for k in (
                                self.dexhand_rh.contact_body_names
                                if side == "rh"
                                else self.dexhand_lh.contact_body_names
                            )
                        ],
                        axis=1,
                    )
                    # 添加力大小
                    tip_force = torch.cat([tip_force, torch.norm(tip_force, dim=-1, keepdim=True)], dim=-1)  
                    pri_obs_values.append(tip_force.reshape(self.num_envs, -1))
                else: # 其他特权观测
                    pri_obs_values.append(side_states[ob])
            obs_dict["privileged"] = torch.cat(pri_obs_values, dim=-1) # 拼接所有特权观测 - 特权观测

        next_target_state = {}

        # 目标idx：获取数据维度并限制索引范围
        nE, nT = side_demo_data["wrist_pos"].shape[:2]
        nF = 1
        
        cur_idx = self.progress_buf
        cur_idx = torch.clamp(cur_idx, min=0, max=nT - 1)  # 限制在有效范围内
        cur_idx = torch.stack([cur_idx + t for t in range(1)], dim=-1)  # [B, K], K = 1

        def indicing(data, idx):
            assert data.shape[0] == nE and data.shape[1] == nT
            remaining_shape = data.shape[2:]
            expanded_idx = idx
            for _ in remaining_shape:
                expanded_idx = expanded_idx.unsqueeze(-1)
            expanded_idx = expanded_idx.expand(-1, -1, *remaining_shape)
            return torch.gather(data, 1, expanded_idx)

        # 目标关节位置 - 当前关节位置 - 维度对齐计算差值
        target_joints_pos = indicing(side_demo_data["mano_joints"], cur_idx).reshape(nE, nF, -1, 3)
        # 应用手动偏移
        target_joints_pos[..., 0] = target_joints_pos[..., 0] + self.manual_offset_dx
        target_joints_pos[..., 1] = target_joints_pos[..., 1] + self.manual_offset_dy
        cur_joint_pos = side_states["joints_state"][:, 1:, :3]  # 跳过手腕
        next_target_state["delta_joints_pos"] = (target_joints_pos - cur_joint_pos[:, None]).reshape(self.num_envs, -1)

        # 获取目标物体位置 - 当前位置 - 维度对齐计算差值
        target_obj_transf = indicing(side_demo_data["obj_trajectory"], cur_idx) # 目标物体变化矩阵
        target_obj_transf = target_obj_transf.reshape(nE * nF, 4, 4) # 形状重塑
        
        # 保存原始旋转矩阵（不应用偏移）
        target_obj_rot = target_obj_transf[:, :3, :3].clone()
        
        # 应用手动偏移到物体位置
        target_obj_pos = target_obj_transf[:, :3, 3].clone()
        # 添加安全检查，确保不会产生 NaN 或无穷大值
        if torch.isnan(target_obj_pos).any() or torch.isinf(target_obj_pos).any():
            print("Warning: target_obj_pos contains NaN or inf values before offset")
            target_obj_pos = torch.zeros_like(target_obj_pos)
        target_obj_pos[:, 0] = target_obj_pos[:, 0] + self.manual_offset_dx
        target_obj_pos[:, 1] = target_obj_pos[:, 1] + self.manual_offset_dy
        # 再次检查偏移后的值
        if torch.isnan(target_obj_pos).any() or torch.isinf(target_obj_pos).any():
            print("Warning: target_obj_pos contains NaN or inf values after offset")
            target_obj_pos = torch.zeros_like(target_obj_pos)
        next_target_state["delta_manip_obj_pos"] = (
            target_obj_pos.reshape(nE, nF, -1) - side_states["manip_obj_pos"][:, None]
        ).reshape(nE, -1)

        # 目标物体四元数 - 使用原始旋转矩阵（不应用偏移）
        # 添加安全检查，确保旋转矩阵有效
        if torch.isnan(target_obj_rot).any() or torch.isinf(target_obj_rot).any():
            print("Warning: target_obj_rot contains NaN or inf values")
            # 使用单位矩阵作为默认值
            target_obj_rot = torch.eye(3, device=target_obj_rot.device, dtype=target_obj_rot.dtype).unsqueeze(0).repeat(target_obj_rot.shape[0], 1, 1)
        next_target_state["manip_obj_quat"] = rotmat_to_quat(target_obj_rot)[:, [1, 2, 3, 0]]
        next_target_state["delta_manip_obj_quat"] = quat_mul(
            side_states["manip_obj_quat"][:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(next_target_state["manip_obj_quat"]),
        ).reshape(nE, -1)
        next_target_state["manip_obj_quat"] = next_target_state["manip_obj_quat"].reshape(nE, -1)

        # 目标物体与关节位置距离 
        next_target_state["obj_to_joints"] = torch.norm(
            side_states["manip_obj_pos"][:, None] - side_states["joints_state"][:, :, :3], dim=-1
        ).reshape(self.num_envs, -1)
        
        # 演示数据中的指尖距离 
        next_target_state["gt_tips_distance"] = indicing(side_demo_data["tips_distance"], cur_idx).reshape(nE, -1)

        # 目标物体bps 
        next_target_state["bps"] = getattr(self, f"obj_bps_{side}")

        # 目标观测 - body link末端的三维空间位置
        obs_dict["target"] = torch.cat(
            [
                next_target_state[ob]
                for ob in [  
                    "delta_joints_pos", # 关节位置差值 num body -1 * 3 
                    "delta_manip_obj_pos", # 物体位置差值 3
                    "manip_obj_quat", # 物体四元数 4
                    "delta_manip_obj_quat", # 物体四元数差值 4
                    "obj_to_joints", # 物体与关节位置距离 num body 
                    "gt_tips_distance", # 演示数据中的指尖距离 5
                    "bps", # 物体bps 128
                ]
            ],
            dim=-1,
        )

        if not self.training:
            manip_obj_root_state = getattr(self, f"_manip_obj_{side}_root_state")
            dexhand_handles = getattr(self, f"dexhand_{side}_handles")
            for prop_name in self._prop_dump_info.keys():
                if prop_name == "state_rh" and side == "rh":
                    # 对于G1机器人，使用手腕位置而不是基座位置
                    self.dump_fileds[prop_name][:] = side_states["joints_state"][:, 0, :]
                elif prop_name == "state_lh" and side == "lh":
                    # 对于G1机器人，使用手腕位置而不是基座位置
                    self.dump_fileds[prop_name][:] = side_states["joints_state"][:, 0, :]
                elif prop_name == "state_manip_obj_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = manip_obj_root_state
                elif prop_name == "state_manip_obj_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = manip_obj_root_state
                elif prop_name == "joint_state_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = torch.stack(
                        [self._rigid_body_state[:, dexhand_handles[k], :] for k in self.dexhand_rh.body_names],
                        dim=1,
                    ).reshape(self.num_envs, -1)
                elif prop_name == "joint_state_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = torch.stack(
                        [self._rigid_body_state[:, dexhand_handles[k], :] for k in self.dexhand_lh.body_names],
                        dim=1,
                    ).reshape(self.num_envs, -1)
                elif prop_name == "q_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = side_states["q"]
                elif prop_name == "q_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = side_states["q"]
                elif prop_name == "dq_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = side_states["dq"]
                elif prop_name == "dq_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = side_states["dq"]
                elif prop_name == "tip_force_rh" and side == "rh":
                    tip_force = torch.stack(
                        [self.net_cf[:, dexhand_handles[k], :] for k in self.dexhand_rh.contact_body_names],
                        axis=1,
                    )
                    self.dump_fileds[prop_name][:] = tip_force.reshape(self.num_envs, -1)
                elif prop_name == "tip_force_lh" and side == "lh":
                    tip_force = torch.stack(
                        [self.net_cf[:, dexhand_handles[k], :] for k in self.dexhand_lh.contact_body_names],
                        axis=1,
                    )
                    self.dump_fileds[prop_name][:] = tip_force.reshape(self.num_envs, -1)
                elif prop_name == "reward":
                    self.dump_fileds[prop_name][:] = self.rew_buf.reshape(self.num_envs, -1).detach()
                else:
                    pass
        return obs_dict  # 用于测试的时候收集数据 -- 做数据集

    def _reset_default(self, env_ids):
        if self.random_state_init:
            if self.rollout_begin is not None:
                seq_idx = (
                    torch.floor(
                        self.rollout_len * 0.98 * torch.rand_like(self.demo_data_rh["seq_len"][env_ids].float())
                    ).long()
                    + self.rollout_begin
                )
                seq_idx = torch.clamp(
                    seq_idx,
                    torch.zeros(1, device=self.device).long(),
                    torch.floor(self.demo_data_rh["seq_len"][env_ids] * 0.98).long(),
                )
            else:
                seq_idx = torch.floor(
                    self.demo_data_rh["seq_len"][env_ids]
                    * 0.98
                    * torch.rand_like(self.demo_data_rh["seq_len"][env_ids].float())
                ).long()
        else:
            if self.rollout_begin is not None:
                seq_idx = self.rollout_begin * torch.ones_like(self.demo_data_rh["seq_len"][env_ids].long())
            else:
                seq_idx = torch.zeros_like(self.demo_data_rh["seq_len"][env_ids].long())

        # 打印第0个环境的起始帧索引和所有重置环境的最大/最小idx
        if 0 in env_ids:
            idx_0 = (env_ids == 0).nonzero(as_tuple=True)[0]
            if len(idx_0) > 0:
                seq_0 = seq_idx[idx_0[0]].item()
                min_seq = seq_idx.min().item()
                max_seq = seq_idx.max().item()
                
                # 构建rollout信息字符串
                if self.rollout_begin is not None and self.rollout_len is not None:
                    rollout_info = f", rollout=[{self.rollout_begin},{self.rollout_begin + self.rollout_len})"
                else:
                    rollout_info = ""
                
                # 获取当前scale_factor
                scale_info = f", scale={self.scale_factor:.4f}" if hasattr(self, 'scale_factor') else ""
                
                # 根据重置原因决定是否显示成功标记
                if hasattr(self, '_env0_reset_reason') and self._env0_reset_reason == "成功完成":
                    print(f"环境0重置(BiH) [成功]: seq_idx={seq_0}, seq_range=[{min_seq},{max_seq}], random={self.random_state_init}{rollout_info}{scale_info}")
                else:
                    print(f"环境0重置(BiH): seq_idx={seq_0}, seq_range=[{min_seq},{max_seq}], random={self.random_state_init}{rollout_info}{scale_info}")

        # 根据数据有效性选择性地重置手部
        if self.use_lh:
            self._reset_default_side(env_ids, seq_idx, side="lh")
        if self.use_rh:
            self._reset_default_side(env_ids, seq_idx, side="rh")

        # 只收集有数据的手的索引
        dexhand_indices = []
        manip_obj_indices = []
        
        if self.use_rh:
            dexhand_indices.append(self._global_dexhand_rh_indices[env_ids].flatten())
            manip_obj_indices.append(self._global_manip_obj_rh_indices[env_ids].flatten())
        
        if self.use_lh:
            dexhand_indices.append(self._global_dexhand_lh_indices[env_ids].flatten())
            manip_obj_indices.append(self._global_manip_obj_lh_indices[env_ids].flatten())
        
        dexhand_multi_env_ids_int32 = torch.concat(dexhand_indices)
        manip_obj_multi_env_ids_int32 = torch.concat(manip_obj_indices)

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(torch.concat([dexhand_multi_env_ids_int32, manip_obj_multi_env_ids_int32])),
            len(torch.concat([dexhand_multi_env_ids_int32, manip_obj_multi_env_ids_int32])),
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )

        self.progress_buf[env_ids] = seq_idx
        self.running_progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.failure_buf[env_ids] = 0
        self.error_buf[env_ids] = 0
        self.total_rew_buf[env_ids] = 0
        self.apply_forces[env_ids] = 0
        self.apply_torque[env_ids] = 0
        self.prev_targets[env_ids] = 0

        self.lh_tips_contact_history[env_ids] = torch.ones_like(self.lh_tips_contact_history[env_ids]).bool()
        self.rh_tips_contact_history[env_ids] = torch.ones_like(self.rh_tips_contact_history[env_ids]).bool()

    def _reset_default_side(self, env_ids, seq_idx, side="rh"):

        side_demo_data = getattr(self, f"demo_data_{side}")

        dof_pos = side_demo_data["opt_dof_pos"][env_ids, seq_idx]
        dof_pos = torch_jit_utils.tensor_clamp(
            dof_pos,
            getattr(self, f"dexhand_{side}_dof_lower_limits").unsqueeze(0),
            getattr(self, f"dexhand_{side}_dof_upper_limits").unsqueeze(0),
        )
        # 初始关节速度设为0
        dof_vel = torch.zeros_like(dof_pos)

        if side == "rh":
            # 完整机器人模式下，右手DOF映射到19-37
            for i, dof_idx in enumerate(self.dexhand_rh.dof_indices):
                self._q[env_ids, dof_idx] = dof_pos[:, i]
                self._qd[env_ids, dof_idx] = dof_vel[:, i]
                self._pos_control[env_ids, dof_idx] = dof_pos[:, i]
        else:
            # 完整机器人模式下，左手DOF映射到0-18
            for i, dof_idx in enumerate(self.dexhand_lh.dof_indices):
                self._q[env_ids, dof_idx] = dof_pos[:, i]
                self._qd[env_ids, dof_idx] = dof_vel[:, i]
                self._pos_control[env_ids, dof_idx] = dof_pos[:, i]

        # reset manip obj
        obj_pos_init = side_demo_data["obj_trajectory"][env_ids, seq_idx, :3, 3]
        # 应用手动偏移到物体位置
        obj_pos_init[:, 0] = obj_pos_init[:, 0] + self.manual_offset_dx
        obj_pos_init[:, 1] = obj_pos_init[:, 1] + self.manual_offset_dy
        obj_rot_init = side_demo_data["obj_trajectory"][env_ids, seq_idx, :3, :3]
        obj_rot_init = rotmat_to_quat(obj_rot_init)
        # [w, x, y, z] to [x, y, z, w]
        obj_rot_init = obj_rot_init[:, [1, 2, 3, 0]]

        obj_vel = torch.zeros_like(side_demo_data["obj_velocity"][env_ids, seq_idx])
        obj_ang_vel = torch.zeros_like(side_demo_data["obj_angular_velocity"][env_ids, seq_idx])

        manip_obj_root_state = getattr(self, f"_manip_obj_{side}_root_state")

        manip_obj_root_state[env_ids, :3] = obj_pos_init
        manip_obj_root_state[env_ids, 3:7] = obj_rot_init
        manip_obj_root_state[env_ids, 7:10] = obj_vel
        manip_obj_root_state[env_ids, 10:13] = obj_ang_vel

    def reset_idx(self, env_ids):
        # 保存环境0的重置原因
        if 0 in env_ids and hasattr(self, 'success_buf'):
            self._env0_reset_reason = "成功完成" if self.success_buf[0].item() > 0 else "其他"
            
        self._refresh()
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

        last_step = self.gym.get_frame_count(self.sim)
        if self.training and len(self.dataIndices) == 1 and last_step >= self.tighten_steps:
            # 初始化 best_rollout_len 和 best_rollout_begin（如果不存在）
            if not hasattr(self, 'best_rollout_len'):
                self.best_rollout_len = 0
                self.best_rollout_begin = 0
            
            running_steps = self.running_progress_buf[env_ids] - 1
            max_running_steps, max_running_idx = running_steps.max(dim=0)
            max_running_env_id = env_ids[max_running_idx]
            if max_running_steps > self.best_rollout_len:
                self.best_rollout_len = max_running_steps
                self.best_rollout_begin = self.progress_buf[max_running_env_id] - 1 - max_running_steps

        self._reset_default(env_ids)

    def reset_done(self):
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)
            self.compute_observations()

        if not self.dict_obs_cls:
            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

            # asymmetric actor-critic
            if self.num_states > 0:
                self.obs_dict["states"] = self.get_state()

        return self.obs_dict, done_env_ids

    def step(self, actions):
        obs, rew, done, info = super().step(actions)
        info["reward_dict"] = self.reward_dict
        info["total_rewards"] = self.total_rew_buf
        info["total_steps"] = self.progress_buf
        return obs, rew, done, info

    def pre_physics_step(self, actions):

        # ? >>> for visualization
        if not self.headless and self.enable_visualization:
            # 目标idx - 限制在有效范围内
            cur_idx = self.progress_buf
            max_idx_rh = self.demo_data_rh["wrist_pos"].shape[1] - 1
            cur_idx = torch.clamp(cur_idx, min=0, max=max_idx_rh)

            self.gym.clear_lines(self.viewer)

            def set_side_joint(cur_idx, side="rh"):
                cur_wrist_pos = getattr(self, f"demo_data_{side}")["wrist_pos"][torch.arange(self.num_envs), cur_idx]
                # 应用手动偏移
                cur_wrist_pos[:, 0] = cur_wrist_pos[:, 0] + self.manual_offset_dx
                cur_wrist_pos[:, 1] = cur_wrist_pos[:, 1] + self.manual_offset_dy
                cur_mano_joint_pos = getattr(self, f"demo_data_{side}")["mano_joints"][
                    torch.arange(self.num_envs), cur_idx
                ].reshape(self.num_envs, -1, 3)
                # 应用手动偏移
                cur_mano_joint_pos[..., 0] = cur_mano_joint_pos[..., 0] + self.manual_offset_dx
                cur_mano_joint_pos[..., 1] = cur_mano_joint_pos[..., 1] + self.manual_offset_dy
                cur_mano_joint_pos = torch.concat([cur_wrist_pos[:, None], cur_mano_joint_pos], dim=1)
                # Visualization disabled to avoid segmentation faults
                # self._init_mano_joint_points()
                # for k in range(len(getattr(self, f"mano_joint_{side}_points"))):
                #     getattr(self, f"mano_joint_{side}_points")[k][:, :3] = cur_mano_joint_pos[:, k]
                for env_id, env_ptr in enumerate(self.envs):
                    for rh_k, k in zip(
                        self.dexhand_rh.body_names,
                        (self.dexhand_rh.body_names if side == "rh" else self.dexhand_lh.body_names),
                    ):
                        self.set_force_vis(
                            env_ptr,
                            rh_k,
                            torch.norm(self.net_cf[env_id, getattr(self, f"dexhand_{side}_handles")[k]], dim=-1) != 0,
                            side,
                        )

                    def add_lines(viewer, env_ptr, hand_joints, color):
                        assert hand_joints.shape[0] == self.dexhand_rh.n_bodies and hand_joints.shape[1] == 3
                        hand_joints = hand_joints.cpu().numpy()
                        lines = np.array([[hand_joints[b[0]], hand_joints[b[1]]] for b in self.dexhand_rh.bone_links])
                        for line in lines:
                            self.gym.add_lines(viewer, env_ptr, 1, line, color)

                    def add_trajectory_lines(viewer, env_ptr, trajectory_points, color):
                        """绘制目标轨迹线条"""
                        if trajectory_points.shape[0] < 2:
                            # print(f"Warning: Not enough trajectory points: {trajectory_points.shape[0]}")  # 注释掉打印
                            return
                        trajectory_points = trajectory_points.cpu().numpy()
                        # print(f"Drawing {trajectory_points.shape[0]} trajectory points")  # 注释掉打印
                        # 创建连接相邻点的线条
                        for i in range(trajectory_points.shape[0] - 1):
                            line = np.array([[trajectory_points[i], trajectory_points[i + 1]]])
                            self.gym.add_lines(viewer, env_ptr, 1, line, color)

                    # 绿色线条 - 目标轨迹
                    if not self.headless and self.enable_visualization:
                        color = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
                        add_trajectory_lines(self.viewer, env_ptr, cur_mano_joint_pos[env_id], color)

            # 获取目标轨迹数据
            active_sides = []
            if self.use_lh:
                active_sides.append("lh")
            if self.use_rh:
                active_sides.append("rh")
            
            for side in active_sides:
                cur_wrist_pos = getattr(self, f"demo_data_{side}")["wrist_pos"][torch.arange(self.num_envs), cur_idx]
                # 应用手动偏移
                cur_wrist_pos[:, 0] = cur_wrist_pos[:, 0] + self.manual_offset_dx
                cur_wrist_pos[:, 1] = cur_wrist_pos[:, 1] + self.manual_offset_dy
                cur_mano_joint_pos = getattr(self, f"demo_data_{side}")["mano_joints"][
                    torch.arange(self.num_envs), cur_idx
                ].reshape(self.num_envs, -1, 3)
                # 应用手动偏移
                cur_mano_joint_pos[..., 0] = cur_mano_joint_pos[..., 0] + self.manual_offset_dx
                cur_mano_joint_pos[..., 1] = cur_mano_joint_pos[..., 1] + self.manual_offset_dy
                cur_mano_joint_pos = torch.concat([cur_wrist_pos[:, None], cur_mano_joint_pos], dim=1)
                
                # 绘制绿色线条
                for env_id, env_ptr in enumerate(self.envs):
                    # 使用已定义的手部骨架连接关系绘制绿色线条
                    trajectory_points = cur_mano_joint_pos[env_id].cpu().numpy() 
                    # Gpu to Cpu 传输
                    
                    
                    ##############################################################
                    if trajectory_points.shape[0] >= 18:  # 确保有足够的关节点
                        # 使用对应手部的骨架连接关系
                        bone_links = getattr(self, f"dexhand_{side}").bone_links
                        
                        color = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
                        for bone in bone_links: #遍历所有骨骼绘制线条
                            if bone[0] < trajectory_points.shape[0] and bone[1] < trajectory_points.shape[0]:
                                line = np.array([[trajectory_points[bone[0]], trajectory_points[bone[1]]]])
                                self.gym.add_lines(self.viewer, env_ptr, 1, line, color)

        res_split_idx = actions.shape[1] // 2

        base_action = actions[:, :res_split_idx]  # ? in the range of [-1, 1]
        residual_action = actions[:, res_split_idx:] * 2  # 残差动作乘以4，增加修正能力

        # 调试：仅首次打印动作形状，避免刷屏
        if not hasattr(self, "_dbg_printed_action_shape"):
            hands_info = []
            if self.use_rh:
                hands_info.append("RH")
            if self.use_lh:
                hands_info.append("LH")
            print(
                f"[ActionShapes] hands={'+'.join(hands_info)}, actions={tuple(actions.shape)}, "
                f"base={tuple(base_action.shape)}, residual={tuple(residual_action.shape)}, "
                f"res_split_idx={res_split_idx}"
            )
            self._dbg_printed_action_shape = True

        # 根据手部配置分配动作
        action_idx = 0
        
        # 右手动作（如果使用右手）
        if self.use_rh:
            rh_dof_pos = (
                1.0 * base_action[:, action_idx:action_idx + self.dexhand_rh.n_dofs]
                + residual_action[:, action_idx:action_idx + self.dexhand_rh.n_dofs]
            )
            rh_dof_pos = torch.clamp(rh_dof_pos, -1, 1)
            action_idx += self.dexhand_rh.n_dofs
        else:
            # 如果不使用右手，保持当前姿态或设置默认姿态
            rh_dof_pos = self.prev_targets[:, :self.dexhand_rh.n_dofs] if hasattr(self, 'prev_targets') else torch.zeros(self.num_envs, self.dexhand_rh.n_dofs, device=self.device)
        
        # 左手动作（如果使用左手）
        if self.use_lh:
            lh_dof_pos = (
                1.0 * base_action[:, action_idx:action_idx + self.dexhand_lh.n_dofs]
                + residual_action[:, action_idx:action_idx + self.dexhand_lh.n_dofs]
            )
            lh_dof_pos = torch.clamp(lh_dof_pos, -1, 1)
        else:
            # 如果不使用左手，保持当前姿态或设置默认姿态
            lh_dof_pos = self.prev_targets[:, self.dexhand_rh.n_dofs:] if hasattr(self, 'prev_targets') else torch.zeros(self.num_envs, self.dexhand_lh.n_dofs, device=self.device)
        
        # 创建完整的38DOF动作向量
        full_dof_targets = torch.zeros(self.num_envs, 38, device=self.device)
        
        # 映射左手DOF (0-18)
        for i, dof_idx in enumerate(self.dexhand_lh.dof_indices):
            full_dof_targets[:, dof_idx] = lh_dof_pos[:, i]
        
        # 映射右手DOF (19-37)
        for i, dof_idx in enumerate(self.dexhand_rh.dof_indices):
            full_dof_targets[:, dof_idx] = rh_dof_pos[:, i]
        
        # 为兼容性设置左右手目标
        self.lh_curr_targets = lh_dof_pos
        self.rh_curr_targets = rh_dof_pos
        self.full_robot_curr_targets = full_dof_targets
    

        curr_act_moving_average = self.act_moving_average

        # 完整机器人的目标设置
        # 左手目标
        self.lh_curr_targets = torch_jit_utils.scale(
            self.lh_curr_targets,
            self.dexhand_lh_dof_lower_limits,
            self.dexhand_lh_dof_upper_limits,
        )
        self.lh_curr_targets = (
            curr_act_moving_average * self.lh_curr_targets
            + (1.0 - curr_act_moving_average) * self.prev_targets[:, : self.dexhand_lh.n_dofs]
        )
        self.lh_curr_targets = torch_jit_utils.tensor_clamp(
            self.lh_curr_targets,
            self.dexhand_lh_dof_lower_limits,
            self.dexhand_lh_dof_upper_limits,
        )
        
        # 右手目标
        self.rh_curr_targets = torch_jit_utils.scale(
            self.rh_curr_targets,
            self.dexhand_rh_dof_lower_limits,
            self.dexhand_rh_dof_upper_limits,
        )
        self.rh_curr_targets = (
            curr_act_moving_average * self.rh_curr_targets
            + (1.0 - curr_act_moving_average) * self.prev_targets[:, self.dexhand_lh.n_dofs : self.dexhand_lh.n_dofs + self.dexhand_rh.n_dofs]
        )
        self.rh_curr_targets = torch_jit_utils.tensor_clamp(
            self.rh_curr_targets,
            self.dexhand_rh_dof_lower_limits,
            self.dexhand_rh_dof_upper_limits,
        )
        
        # 添加肩部关节硬约束
        self.lh_curr_targets[:, 1] = torch.clamp(self.lh_curr_targets[:, 1], min=0.0)
        self.rh_curr_targets[:, 1] = torch.clamp(self.rh_curr_targets[:, 1], max=-0.0)
        self.lh_curr_targets[:, 0] = torch.clamp(self.lh_curr_targets[:, 0], min=-1.0)
        self.rh_curr_targets[:, 0] = torch.clamp(self.rh_curr_targets[:, 0], min=-1.0)
        
        # 更新完整机器人目标（应用约束后）
        for i, dof_idx in enumerate(self.dexhand_lh.dof_indices):
            self.full_robot_curr_targets[:, dof_idx] = self.lh_curr_targets[:, i]
        for i, dof_idx in enumerate(self.dexhand_rh.dof_indices):
            self.full_robot_curr_targets[:, dof_idx] = self.rh_curr_targets[:, i]

        # 更新prev_targets
        self.prev_targets[:, : self.dexhand_lh.n_dofs] = self.lh_curr_targets[:]
        self.prev_targets[:, self.dexhand_lh.n_dofs : self.dexhand_lh.n_dofs + self.dexhand_rh.n_dofs] = self.rh_curr_targets[:]

        # 设置最终目标
        self._pos_control[:] = self.full_robot_curr_targets[:]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

    def post_physics_step(self):

        self.compute_observations()
        self.compute_reward(self.actions)

        self.progress_buf += 1
        self.running_progress_buf += 1
        self.randomize_buf += 1

    def create_camera(
        self,
        *,
        env,
        isaac_gym,
    ):
        """
        Only create front camera for view purpose
        """
        if self._record:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 1280
            camera_cfg.height = 720
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.80, -0.00, 0.7)
            cam_target = gymapi.Vec3(-1, -0.00, 0.3)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        else:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 320
            camera_cfg.height = 180
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.5)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        return camera

    def set_force_vis(self, env_ptr, part_k, has_force, side):
        self.gym.set_rigid_body_color(
            env_ptr,
            self.gym.find_actor_handle(env_ptr, "full_robot"),
            getattr(self, f"dexhand_rh_handles")[part_k],  # tricks here, because the handle is the same
            gymapi.MESH_VISUAL,
            (
                gymapi.Vec3(
                    1.0,
                    0.6,
                    0.6,
                )
                if has_force
                else gymapi.Vec3(1.0, 1.0, 1.0)
            ),
        )

    def update_success_rate(self, success_rate):
        """更新最近成功率，用于动态奖励调整
        
        Args:
            success_rate: 最近的成功率 [0.0, 1.0]
        """
        self.recent_success_rate = success_rate if success_rate is not None else 0.0


@torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis


@torch.jit.script
def compute_imitation_reward(
    reset_buf: Tensor,
    progress_buf: Tensor,
    running_progress_buf: Tensor,
    actions: Tensor,
    states: Dict[str, Tensor],
    target_states: Dict[str, Tensor],
    max_length: List[int],
    scale_factor: float,
    success_rate: float,
    dexhand_weight_idx: Dict[str, List[int]],
    side: str,
    rigid_body_state: Tensor,
    dof_acc: Tensor,
    action_rate: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:

    # ==================================================================================
    # 奖励计算 (SkillMimic: r = rh × ro × rrel × rreg × rcg)
    # ==================================================================================
    
    current_dof_pos = states["q"]   # 可控关节位置
    current_dof_vel = states["dq"]  # 可控关节速度
    
    # ========== 1. 关节位置奖励 ==========
    joints_pos = states["joints_state"][:, 1:, :3] 
    target_joints_pos = target_states["joints_pos"]
    diff_joints_pos = target_joints_pos - joints_pos
    diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)
    
    # 提取各个关节的位置误差
    diff_thumb_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["thumb_tip"]]].mean(dim=-1)
    diff_index_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["index_tip"]]].mean(dim=-1)
    diff_middle_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["middle_tip"]]].mean(dim=-1)
    diff_ring_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["ring_tip"]]].mean(dim=-1)
    diff_pinky_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["pinky_tip"]]].mean(dim=-1)
    diff_level_1_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_1_joints"]]].mean(dim=-1)
    diff_level_2_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_2_joints"]]].mean(dim=-1)
    
    lambda_p = 20.0
    reward_thumb_tip_pos = torch.exp(-20 * diff_thumb_tip_pos_dist)
    reward_index_tip_pos = torch.exp(-20 * diff_index_tip_pos_dist)
    reward_middle_tip_pos = torch.exp(-15 * diff_middle_tip_pos_dist)
    reward_pinky_tip_pos = torch.exp(-10 * diff_pinky_tip_pos_dist)
    reward_ring_tip_pos = torch.exp(-5 * diff_ring_tip_pos_dist)
    reward_level_1_pos = torch.exp(-5 * diff_level_1_pos_dist)
    reward_level_2_pos = torch.exp(-5 * diff_level_2_pos_dist)
    
    # ========== 2. 物体运动学奖励 ==========
    current_obj_pos = states["manip_obj_pos"] 
    current_obj_quat = states["manip_obj_quat"]
    target_obj_pos = target_states["manip_obj_pos"]
    target_obj_quat = target_states["manip_obj_quat"]
    
    # 物体位置奖励
    diff_obj_pos = target_obj_pos - current_obj_pos 
    diff_obj_pos_dist = torch.norm(diff_obj_pos, dim=-1)
    lambda_op = 20.0
    reward_obj_pos = torch.exp(-lambda_op * diff_obj_pos_dist)
    
    # 物体旋转奖励
    # diff_obj_rot = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))
    # diff_obj_rot_angle = quat_to_angle_axis(diff_obj_rot)[0]
    # lambda_or = 20.0
    # reward_obj_rot = torch.exp(-lambda_or * diff_obj_rot_angle.abs())
    
    # ========== 3. 相对运动奖励 ==========
    current_relative_pos = joints_pos - current_obj_pos[:, None, :]  # [N, n_joints, 3]
    target_relative_pos = target_states["relative_pos"]  # [N, n_joints, 3]
    diff_relative_pos = target_relative_pos - current_relative_pos
    diff_relative_pos_dist = torch.norm(diff_relative_pos, dim=-1)  # [N, n_joints]
    mean_relative_error = diff_relative_pos_dist.mean(dim=-1)  # [N]
    
    lambda_rel = 20.0
    reward_relative_motion = torch.exp(-lambda_rel * mean_relative_error)
    
    # ========== 4. 正则化奖励 ========== 数值微分计算从dof位置到dof速度 - 再高斯平滑处理 - 都是手指加手臂19个DOF
    target_dof_vel = target_states["dof_vel"]
    dof_acc_squared = torch.norm(dof_acc, dim=-1) ** 2
    target_dof_vel_squared = torch.norm(target_dof_vel, dim=-1) ** 2
    lambda_reg = 1e-6 # 1e-12可能太小了 - 1e-8可以试试 - 1e-6可以
    e_acc = dof_acc_squared / (target_dof_vel_squared + lambda_reg)
    reward_velocity_reg = torch.exp(-lambda_reg * e_acc)

    reward_acc = torch.exp(-1e-3 * torch.norm(dof_acc, dim=-1))
    reward_action_rate = torch.exp(-20 * torch.norm(action_rate, dim=-1))

    # ========== 5. 接触图奖励 ==========
    contact_graph_current = target_states["contact_graph_current"]  # [N, 5]
    contact_graph_target = target_states["contact_graph_target"]    # [N, 5]
    contact_graph_error = torch.abs(contact_graph_current - contact_graph_target)  # [N, 5]
    total_cg_error = contact_graph_error.sum(dim=-1)  # [N] - 5个指尖的接触误差总和
    
    lambda_cg = 5.0
    reward_contact_graph = torch.exp(-lambda_cg * total_cg_error)
    
    # ========== 额外奖励：接触力奖励 ==========
    finger_tip_force = target_states["tip_force"]
    finger_tip_distance = target_states["tips_distance"]
    contact_range = [0.02, 0.03]
    finger_tip_weight = torch.clamp(
        (contact_range[1] - finger_tip_distance) / (contact_range[1] - contact_range[0]), 0, 1
    )
    finger_tip_force_masked = finger_tip_force * finger_tip_weight[:, :, None]
    finger_force_magnitudes = torch.norm(finger_tip_force_masked, dim=-1)
    reward_finger_tip_force = torch.exp(-1 * (1 / (finger_force_magnitudes.sum(-1) + 1e-5)))
    
    # ========== 惩罚项：肩部关节约束 ==========
    shoulder_roll_pos = current_dof_pos[:, 1] 
    shoulder_pitch_pos = current_dof_pos[:, 0] 
    if side == "lh":
        penalty_shoulder_roll = torch.clamp(0.0 - shoulder_roll_pos, min=0.0)
    else:
        penalty_shoulder_roll = torch.clamp(shoulder_roll_pos + 0.0, min=0.0)
    
    penalty_shoulder_pitch = torch.clamp(-1.0 - shoulder_pitch_pos, min=0.0)

    threshold_obj = 0.02 / 0.343 * scale_factor**3  # 进一步放宽物体位置阈值  0.343 * scale_factor**3
    # threshold_rot = 30.0 / 0.343 * scale_factor**3   # 进一步放宽旋转阈值 
    threshold_finger_thumb = 0.04 / 0.7 * scale_factor    # 进一步放宽拇指阈值
    threshold_finger_index = 0.045 / 0.7 * scale_factor   # 进一步放宽食指阈值
    threshold_finger_middle = 0.05 / 0.7 * scale_factor   # 进一步放宽中指阈值
    threshold_finger_pinky = 0.06 / 0.7 * scale_factor    # 进一步放宽小指阈值
    threshold_finger_ring = 0.06 / 0.7 * scale_factor     # 进一步放宽无名指阈值
    threshold_level1 = 0.07 / 0.7 * scale_factor
    threshold_level2 = 0.08 / 0.7 * scale_factor

    deviation_fail = (
            (diff_obj_pos_dist > threshold_obj) 
            # | (diff_obj_rot_angle.abs() / np.pi * 180 > threshold_rot)  
            | (diff_thumb_tip_pos_dist > threshold_finger_thumb)
            | (diff_index_tip_pos_dist > threshold_finger_index)
            | (diff_middle_tip_pos_dist > threshold_finger_middle)
            | (diff_pinky_tip_pos_dist > threshold_finger_pinky)
            | (diff_ring_tip_pos_dist > threshold_finger_ring)
            | (diff_level_1_pos_dist > threshold_level1)
            | (diff_level_2_pos_dist > threshold_level2)
            | torch.any((finger_tip_distance < 0.005) & ~(target_states["tip_contact_state"].any(1)), dim=-1)
        )
    time_gate = running_progress_buf >= 10
    failed_execute = (deviation_fail & time_gate)
    actual_task_progress = torch.clamp(running_progress_buf, min=0).float()
    
    # ========== 奖励分类组合 (SkillMimic: r = rh × ro × rrel × rreg × rcg) ==========
    
    # 1.关节位置奖励 (内部用乘法)
    reward_joints = (
        reward_thumb_tip_pos
        * reward_index_tip_pos
        * reward_middle_tip_pos
        * reward_pinky_tip_pos
        * reward_ring_tip_pos
        * reward_level_1_pos
        * reward_level_2_pos
    )
    
    # 2.物体运动学奖励 (内部用乘法)
    reward_object = (
        reward_obj_pos   # 物体位置
        # * reward_obj_rot  # 物体旋转
    )
    
    # 3.相对运动奖励
    reward_relative = reward_relative_motion
    
    # 4.速度正则化奖励
    reward_reg = reward_velocity_reg
    
    # 5.接触图奖励
    reward_cg = reward_contact_graph
    
    # ========== 总奖励 (乘法形式，符合论文) ==========
    reward_execute = (
        reward_joints          # rh - 关节位置
        * reward_object        # ro - 物体运动学
        * reward_relative      # rrel - 相对运动
        * reward_reg           # rreg - 速度正则化
        * reward_cg            # rcg - 接触图
        * reward_action_rate   # 动作平滑（正则化）
        * reward_acc           # 加速度正则化
    )
    
    # ========== 额外的惩罚项（加法形式） ==========
    reward_execute = (
        reward_execute
        - 1.0 * penalty_shoulder_roll    # 肩部roll约束
        - 1.0 * penalty_shoulder_pitch   # 肩部pitch约束
    ) 
    
    lambda_scale = (1.0 - scale_factor) / 0.3 
    reward_scale_multiplier = 1.0 + lambda_scale ** 0.5  
    reward_execute = reward_execute * reward_scale_multiplier

    succeeded = (
        progress_buf >= max_length[0] * 1.0
    ) & ~failed_execute  
    
    reset_buf = torch.where(
        succeeded | failed_execute,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    
    reward_dict = {
        # ========== 详细奖励项 ==========
        # 物体奖励
        "reward_obj_pos": reward_obj_pos,
        # "reward_obj_rot": reward_obj_rot,
        # 相对运动
        "reward_relative_motion": reward_relative_motion,
        # 关节位置（每个手指分开）
        "reward_thumb_tip_pos": reward_thumb_tip_pos,
        "reward_index_tip_pos": reward_index_tip_pos,
        "reward_middle_tip_pos": reward_middle_tip_pos,
        "reward_pinky_tip_pos": reward_pinky_tip_pos,
        "reward_ring_tip_pos": reward_ring_tip_pos,
        "reward_level_1_pos": reward_level_1_pos,
        "reward_level_2_pos": reward_level_2_pos,
        # 其他
        "reward_velocity_reg": reward_velocity_reg,
        "reward_contact_graph": reward_contact_graph,
        
        # ========== 额外项 ==========
        "penalty_shoulder_roll": penalty_shoulder_roll,
        "penalty_shoulder_pitch": penalty_shoulder_pitch,
        "reward_action_rate": reward_action_rate,  # 动作平滑奖励
        "reward_acc": reward_acc,  # 加速度奖励
        # "reward_finger_tip_force": reward_finger_tip_force,
        
        # ========== 误差信息（用于统计） ==========
        "error_thumb_tip_pos_dist": diff_thumb_tip_pos_dist,
        "error_level_1_pos_dist": diff_level_1_pos_dist,
        "error_obj_pos_dist": diff_obj_pos_dist,
        
        # ========== 调试信息 ==========
        "dbg_actual_task_progress": actual_task_progress,
        "dbg_max_length0": torch.full_like(progress_buf.float(), float(max_length[0])),
    }

    return reward_execute, reset_buf, succeeded, failed_execute, reward_dict
