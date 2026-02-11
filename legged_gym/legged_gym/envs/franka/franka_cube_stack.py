# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""
FrankaCubeStack: Cube stacking environment for Franka arm in legged_gym.

The task involves a Franka arm picking up cubeA and stacking it on top of cubeB.

Features:
- OSC (Operational Space Control) and joint torque control modes
- Cube assets (cubeA to pick, cubeB as base)
- Table and table stand creation
- Shaped rewards for reaching, lifting, aligning, and stacking
- Collision-free cube reset logic
"""

import os
import numpy as np
import torch
from typing import Dict, Tuple
import ipdb

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import to_torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.helpers import class_to_dict
from .franka_cube_stack_config import FrankaCubeStackCfg


def tensor_clamp(x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    """Clamp tensor values element-wise between min and max tensors."""
    return torch.max(torch.min(x, max_val), min_val)


def axisangle2quat(vec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Converts scaled axis-angle to quaternion.
    
    Args:
        vec: (..., 3) tensor where final dim is (ax, ay, az) axis-angle exponential coordinates
        eps: Stability value below which small values will be mapped to 0

    Returns:
        (..., 4) tensor where final dim is (x, y, z, w) quaternion
    """
    # Store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero and convert to quaternion
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class FrankaCubeStack(BaseTask):
    """
    Cube stacking environment for Franka arm.
    
    The robot must pick up cubeA and stack it on top of cubeB.
    
    Observations (19 dim for OSC, 26 for joint_tor):
        - cubeA_quat (4): Cube A orientation
        - cubeA_pos (3): Cube A position
        - cubeA_to_cubeB_pos (3): Vector from cube A to cube B
        - eef_pos (3): End-effector position
        - eef_quat (4): End-effector orientation
        - q_gripper (2) for OSC mode, or q (9) for joint_tor mode
        
    Rewards:
        - dist: Distance from gripper to cube A
        - lift: Bonus for lifting cube A
        - align: Alignment of cube A above cube B
        - stack: Successful stacking bonus
    
    Attributes:
        cfg: FrankaCubeStackCfg configuration object
        states: Dictionary of current state tensors
        handles: Dictionary of rigid body handles
    """

    def __init__(
        self, 
        cfg: FrankaCubeStackCfg, 
        sim_params, 
        physics_engine, 
        sim_device: str, 
        headless: bool
    ):
        """
        Initialize FrankaCubeStack environment.
        
        Args:
            cfg: Environment configuration
            sim_params: Simulation parameters from IsaacGym
            physics_engine: Physics engine (PhysX)
            sim_device: Device string ('cuda:0' or 'cpu')
            headless: Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.debug_viz = getattr(self.cfg.viewer, "debug_viz", False)
        self.init_done = False
        
        # Store cube config before parent init (since _create_envs needs it)
        self.cubeA_size = cfg.cubes.cubeA_size
        self.cubeB_size = cfg.cubes.cubeB_size
        
        # Parse configuration
        self._parse_cfg(self.cfg)
        
        # Validate control type
        assert self.cfg.control.control_type in {"osc", "joint_tor"}, \
            f"Invalid control type: {self.cfg.control.control_type}. Must be 'osc' or 'joint_tor'"
        
        # Update num_observations and num_actions based on control type
        if self.cfg.control.control_type == "osc":
            # OSC: cubeA_quat(4) + cubeA_pos(3) + cubeA_to_cubeB_pos(3) + eef_pos(3) + eef_quat(4) + q_gripper(2) = 19
            self.cfg.env.num_observations = 19
            # OSC actions: delta EEF pose (6) + gripper (1) = 7
            self.cfg.env.num_actions = 7
        else:
            # Joint torque: same obs but with full q instead of q_gripper = 26
            self.cfg.env.num_observations = 26
            # Joint torque actions: joint torques (7) + gripper (1) = 8
            self.cfg.env.num_actions = 8
        
        # State dictionaries
        self.states: Dict[str, torch.Tensor] = {}
        self.handles: Dict[str, int] = {}
        
        # Tensor placeholders (initialized in _init_buffers)
        self._root_state = None             # State of root body (n_envs, n_actors, 13)
        self._dof_state = None              # State of all joints (n_envs, n_dof, 2)
        self._q = None                      # Joint positions (n_envs, n_dof)
        self._qd = None                     # Joint velocities (n_envs, n_dof)
        self._rigid_body_state = None       # State of all rigid bodies (n_envs, n_bodies, 13)
        self._eef_state = None              # End effector state (at grip site)
        self._eef_lf_state = None           # Left fingertip state
        self._eef_rf_state = None           # Right fingertip state
        self._cubeA_state = None            # Cube A state
        self._cubeB_state = None            # Cube B state
        self._j_eef = None                  # Jacobian for end effector
        self._mm = None                     # Mass matrix
        self._arm_control = None            # Tensor buffer for arm control
        self._gripper_control = None        # Tensor buffer for gripper control
        self._pos_control = None            # Position control actions
        self._effort_control = None         # Torque/effort control actions
        self._franka_effort_limits = None   # Actuator effort limits
        self._global_indices = None         # Unique indices for all actors in flattened array
        
        # Values filled during env creation
        self.num_dofs = None
        self.actions = None
        
        # Initialize base task (calls create_sim)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        
        # Set up camera if not headless
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        
        # Initialize buffers and prepare rewards
        self._init_buffers()
        self._prepare_reward_function()
        
        # Set default DOF positions from config
        default_angles = self.cfg.init_state.default_joint_angles
        self.franka_default_dof_pos = to_torch([
            default_angles["panda_joint1"],
            default_angles["panda_joint2"],
            default_angles["panda_joint3"],
            default_angles["panda_joint4"],
            default_angles["panda_joint5"],
            default_angles["panda_joint6"],
            default_angles["panda_joint7"],
            default_angles["panda_finger_joint1"],
            default_angles["panda_finger_joint2"],
        ], device=self.device)
        
        # OSC gains
        self.kp = to_torch(self.cfg.control.kp, device=self.device)
        self.kd = self.cfg.control.kd_factor * torch.sqrt(self.kp)
        self.kp_null = to_torch(self.cfg.control.kp_null, device=self.device)
        self.kd_null = self.cfg.control.kd_null_factor * torch.sqrt(self.kp_null)
        
        # Set control limits
        if self.cfg.control.control_type == "osc":
            self.cmd_limit = to_torch(self.cfg.control.osc_cmd_limit, device=self.device).unsqueeze(0)
        else:
            self.cmd_limit = self._franka_effort_limits[:7].unsqueeze(0)
        
        # Store reward thresholds
        self.lift_height_threshold = cfg.rewards.lift_height_threshold
        self.stack_xy_threshold = cfg.rewards.stack_xy_threshold
        self.stack_z_threshold = cfg.rewards.stack_z_threshold
        self.gripper_away_threshold = cfg.rewards.gripper_away_threshold
        
        # Initial cube height (for lift detection)
        self._cubeA_initial_height = self._table_surface_pos[2] + self.cubeA_size / 2.0
        
        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
        # Initial tensor refresh
        self._refresh()
        # ipdb.set_trace()
        self.init_done = True

    # -------- Simulation Setup --------

    def create_sim(self):
        """Create simulation with ground plane and environments."""
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        
        self.up_axis_idx = 2  # Z-up
        self.sim = self.gym.create_sim(
            self.sim_device_id, 
            self.graphics_device_id, 
            self.physics_engine, 
            self.sim_params
        )
        
        self._create_ground_plane()
        self._create_envs()

    def _create_ground_plane(self):
        """Create ground plane for the simulation."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """
        Create environments with Franka arm, table, table stand, and cubes.
        
        This method:
        1. Loads Franka URDF with appropriate DOF settings
        2. Creates table and table stand as box assets
        3. Creates cubeA and cubeB assets
        4. Sets up DOF properties for arm (effort control) and gripper (position control)
        5. Creates environments in a grid layout
        """
        spacing = self.cfg.env.env_spacing
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        # Load Franka asset
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = self.cfg.asset.use_mesh_materials
        # asset_options.angular_damping = self.cfg.asset.angular_damping
        # asset_options.linear_damping = self.cfg.asset.linear_damping
        # asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        # asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        # asset_options.armature = self.cfg.asset.armature
        
        franka_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        
        print(f"Number of Franka bodies: {self.num_franka_bodies}")
        print(f"Number of Franka DOFs: {self.num_franka_dofs}")
        
        # DOF stiffness and damping
        franka_dof_stiffness = to_torch(self.cfg.asset.dof_stiffness, dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch(self.cfg.asset.dof_damping, dtype=torch.float, device=self.device)
        
        # Create table asset
        table_pos = self.cfg.table.pos
        table_size = self.cfg.table.size
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_size[0], table_size[1], table_size[2], table_opts)
        
        # Create table stand asset
        stand_height = self.cfg.table.stand_height
        stand_size = self.cfg.table.stand_size
        stand_offset = self.cfg.table.stand_offset
        table_stand_pos = [
            table_pos[0] + stand_offset[0], 
            table_pos[1] + stand_offset[1], 
            table_pos[2] + table_size[2] / 2 + stand_height / 2
        ]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, stand_size[0], stand_size[1], stand_height, table_stand_opts)
        
        # Store table surface position
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_size[2] / 2])
        
        # Create cube assets
        cubeA_opts = gymapi.AssetOptions()
        cubeA_opts.density = 400.0  # kg/m^3
        cubeA_asset = self.gym.create_box(self.sim, self.cubeA_size, self.cubeA_size, self.cubeA_size, cubeA_opts)
        
        cubeB_opts = gymapi.AssetOptions()
        cubeB_opts.density = 400.0
        cubeB_asset = self.gym.create_box(self.sim, self.cubeB_size, self.cubeB_size, self.cubeB_size, cubeB_opts)
        
        # Store cube colors for later application
        self._cubeA_color = gymapi.Vec3(*self.cfg.cubes.cubeA_color)
        self._cubeB_color = gymapi.Vec3(*self.cfg.cubes.cubeB_color)
        
        # Set Franka DOF properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        
        for i in range(self.num_franka_dofs):
            # Arm joints use effort control, gripper uses position control
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i].item()
                franka_dof_props['damping'][i] = franka_dof_damping[i].item()
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0
            
            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])
        
        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        
        # Override gripper effort limits
        franka_dof_props['effort'][7] = self.cfg.asset.gripper_effort_limit
        franka_dof_props['effort'][8] = self.cfg.asset.gripper_effort_limit
        
        # Speed scales for DOFs
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = self.cfg.asset.gripper_speed_scale
        
        # Define start poses
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(
            self.cfg.init_state.pos[0] + stand_offset[0],
            self.cfg.init_state.pos[1] + stand_offset[1],
            table_pos[2] + table_size[2] / 2 + stand_height + self.cfg.init_state.pos[2]
        )
        franka_start_pose.r = gymapi.Quat(*self.cfg.init_state.rot)
        
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        # Cube start poses (will be randomized at reset)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(
            self._table_surface_pos[0],
            self._table_surface_pos[1],
            self._table_surface_pos[2] + self.cubeA_size / 2.0
        )
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        cubeB_start_pose = gymapi.Transform()
        cubeB_start_pose.p = gymapi.Vec3(
            self._table_surface_pos[0],
            self._table_surface_pos[1],
            self._table_surface_pos[2] + self.cubeB_size / 2.0
        )
        cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        # Aggregate mode settings (include cubes)
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 4  # table, table stand, cubeA, cubeB
        max_agg_shapes = num_franka_shapes + 4
        
        self.frankas = []
        self.envs = []
        self.cubeA_idxs = []
        self.cubeB_idxs = []
        
        num_per_row = int(np.sqrt(self.num_envs))
        
        # Create environments
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            if self.cfg.env.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            
            # Create Franka with optional position/rotation noise
            current_franka_pose = gymapi.Transform()
            current_franka_pose.p = franka_start_pose.p
            current_franka_pose.r = franka_start_pose.r
            
            if self.cfg.domain_rand.franka_position_noise > 0:
                rand_xy = self.cfg.domain_rand.franka_position_noise * (-1.0 + np.random.rand(2) * 2.0)
                current_franka_pose.p.x += rand_xy[0]
                current_franka_pose.p.y += rand_xy[1]
            
            if self.cfg.domain_rand.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.cfg.domain_rand.franka_rotation_noise * (-1.0 + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                current_franka_pose.r = gymapi.Quat(*new_quat)
            
            franka_actor = self.gym.create_actor(
                env_ptr, franka_asset, current_franka_pose, "franka", i, 0, 0
            )
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)
            
            if self.cfg.env.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            
            # Create table and table stand
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(
                env_ptr, table_stand_asset, table_stand_start_pose, "table_stand", i, 1, 0
            )
            
            if self.cfg.env.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            
            # Create cubes
            # Actor index 3 = cubeA, index 4 = cubeB
            cubeA_actor = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)
            cubeB_actor = self.gym.create_actor(env_ptr, cubeB_asset, cubeB_start_pose, "cubeB", i, 4, 0)
            
            # Set cube colors
            self.gym.set_rigid_body_color(env_ptr, cubeA_actor, 0, gymapi.MESH_VISUAL, self._cubeA_color)
            self.gym.set_rigid_body_color(env_ptr, cubeB_actor, 0, gymapi.MESH_VISUAL, self._cubeB_color)
            
            if self.cfg.env.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
            
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            
            # Store global cube indices
            self.cubeA_idxs.append(self.gym.get_actor_index(env_ptr, cubeA_actor, gymapi.DOMAIN_SIM))
            self.cubeB_idxs.append(self.gym.get_actor_index(env_ptr, cubeB_actor, gymapi.DOMAIN_SIM))
        
        # Convert cube indices to tensors
        self.cubeA_idxs = to_torch(self.cubeA_idxs, dtype=torch.long, device=self.device)
        self.cubeB_idxs = to_torch(self.cubeB_idxs, dtype=torch.long, device=self.device)
        
        # Initialize data after all environments are created
        self._init_data()

    def _init_data(self):
        """Initialize simulation handles and tensor buffers after environment creation."""
        env_ptr = self.envs[0]
        franka_handle = 0
        
        # Setup rigid body handles
        self.handles = {
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_grip_site"),
        }
        
        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        
        # Acquire tensor handles
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        # Wrap tensors
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        
        # Extract position and velocity
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        
        # End-effector states
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        
        # Cube states from root state tensor
        # Actors order: 0=franka, 1=table, 2=table_stand, 3=cubeA, 4=cubeB
        self._cubeA_state = self._root_state[:, 3, :]
        self._cubeB_state = self._root_state[:, 4, :]
        
        # Jacobian for end-effector
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        
        # Mass matrix
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]
        
        # Initialize control tensors
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)
        
        # Arm uses effort control, gripper uses position control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]
        
        # Number of actors per environment: franka, table, table_stand, cubeA, cubeB
        self.num_actors_per_env = 5
        
        # Global indices for indexed tensor updates
        self._global_indices = torch.arange(
            self.num_envs * self.num_actors_per_env, 
            dtype=torch.int32, 
            device=self.device
        ).view(self.num_envs, -1)
        
        # Initial cube positions (for collision-free sampling)
        self._init_cubeA_state = self._cubeA_state.clone()
        self._init_cubeB_state = self._cubeB_state.clone()
        # import ipdb; ipdb.set_trace()
    #----------------------------------------
    def _init_buffers(self):
        """Initialize additional tensor buffers used during training."""
        # Common step counter
        self.common_step_counter = 0
        
        # Extras for logging
        self.extras = {}
        
        # Action buffer
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros_like(self.actions)

    def _prepare_reward_function(self):
        """
        Prepare reward functions based on configuration.
        
        Looks for self._reward_<REWARD_NAME> methods where <REWARD_NAME>
        matches non-zero reward scales in the config.
        """
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        
        # Remove zero scales and multiply non-zero by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        
        # Prepare list of reward functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            func_name = '_reward_' + name
            if hasattr(self, func_name):
                self.reward_functions.append(getattr(self, func_name))
                print(f"Registered reward function: {func_name}")
            else:
                print(f"Warning: Reward function {func_name} not found")
        
        # Episode sums for logging
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    def _parse_cfg(self, cfg):
        """Parse configuration and set derived values."""
        self.dt = cfg.control.decimation * cfg.sim.dt
        self.max_episode_length_s = cfg.env.episode_length_s
        self.max_episode_length = int(np.ceil(self.max_episode_length_s / self.dt))

    def set_camera(self, position, lookat):
        """Set camera position and direction."""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # -------- State Management --------

    def _update_states(self):
        """Update state dictionary with current tensor values."""
        self.states.update({
            # Franka states
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "qd": self._qd[:, :],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
            # Cube states
            "cubeA_pos": self._cubeA_state[:, :3],
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_vel": self._cubeA_state[:, 7:10],
            "cubeA_ang_vel": self._cubeA_state[:, 10:13],
            "cubeB_pos": self._cubeB_state[:, :3],
            "cubeB_quat": self._cubeB_state[:, 3:7],
            "cubeB_vel": self._cubeB_state[:, 7:10],
            "cubeB_ang_vel": self._cubeB_state[:, 10:13],
            # Relative positions
            "cubeA_to_cubeB_pos": self._cubeB_state[:, :3] - self._cubeA_state[:, :3],
        })

    def _refresh(self):
        """Refresh tensor states from simulation."""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        
        # Update state dictionary
        self._update_states()

    # -------- Control --------

    def _compute_osc_torques(self, dpose: torch.Tensor) -> torch.Tensor:
        """
        Compute Operational Space Control (OSC) torques.
        
        Implements the OSC controller from:
        - Khatib, 1987: https://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        - Tutorial: https://studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        
        Args:
            dpose: Desired change in end-effector pose (6D: dx, dy, dz, droll, dpitch, dyaw)
        
        Returns:
            Joint torques for the 7 arm joints
        """
        q, qd = self._q[:, :7], self._qd[:, :7]
        
        # Regularization for numerical stability
        reg = 1e-6 * torch.eye(7, device=self.device).unsqueeze(0)
        reg6 = 1e-6 * torch.eye(6, device=self.device).unsqueeze(0)
        
        # Compute inverse of mass matrix with regularization
        mm_inv = torch.inverse(self._mm + reg)
        
        # Compute end-effector inertia matrix inverse and forward
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv + reg6)
        
        # Get EEF velocity
        eef_vel = self.states["eef_vel"]
        
        # Debug: Check for NaN in intermediate values
        if torch.any(~torch.isfinite(eef_vel)):
            print("NaN/Inf in eef_vel!")
            # ipdb.set_trace()
        
        # Transform cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
            self.kp * dpose - self.kd * eef_vel
        ).unsqueeze(-1)
        
        # Nullspace control torques prevent large changes in joint configuration
        # Added into the nullspace of OSC so end-effector orientation remains constant
        # Reference: http://roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
            (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi
        )
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null
        
        # Debug: Check for NaN in torque output
        if torch.any(~torch.isfinite(u)):
            print("NaN/Inf in computed torques!")
            print(f"  mm_inv has NaN: {torch.any(~torch.isfinite(mm_inv))}")
            print(f"  m_eef has NaN: {torch.any(~torch.isfinite(m_eef))}")
            print(f"  j_eef has NaN: {torch.any(~torch.isfinite(self._j_eef))}")
            print(f"  dpose has NaN: {torch.any(~torch.isfinite(dpose))}")
            # ipdb.set_trace()
        
        # Clip to valid effort range
        u = tensor_clamp(
            u.squeeze(-1),
            -self._franka_effort_limits[:7].unsqueeze(0),
            self._franka_effort_limits[:7].unsqueeze(0)
        )
        
        return u

    # -------- Step Functions --------

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Execute one environment step.
        
        Args:
            actions: Actions from policy (num_envs, num_actions)
        
        Returns:
            Tuple of (obs, privileged_obs, rewards, dones, info)
        """
        self.pre_physics_step(actions)
        # ipdb.set_trace()
        # Render
        self.render()
        
        # Simulate physics
        for _ in range(self.cfg.control.decimation):
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            
        
        # # Sync GPU simulation results before reading state (required when headless on GPU)
        # if self.device != 'cpu':
        #     self.gym.fetch_results(self.sim, True)
        
        self.post_physics_step()
        
        # Clip observations
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def pre_physics_step(self, actions: torch.Tensor):
        """
        Process actions before physics simulation.
        
        Handles both OSC and joint torque control modes, plus gripper control.
        
        Args:
            actions: Raw actions from policy
        """
        # clip_actions = self.cfg.normalization.clip_actions
        # self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        
        # # Split arm and gripper commands
        # u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]
        
        # # Control arm (scale and apply control type)
        # u_arm = u_arm * self.cmd_limit / self.cfg.control.action_scale
        
        # if self.cfg.control.control_type == "osc":
        #     # OSC mode: Convert 6D delta pose to 7 joint torques via operational space control
        #     u_arm = self._compute_osc_torques(dpose=u_arm)
        # else:
        #     # joint_tor mode: Actions are already 7 joint torques, scaled by effort limits
        #     # Clip to effort limits for safety
        #     u_arm = tensor_clamp(
        #         u_arm,
        #         -self._franka_effort_limits[:7].unsqueeze(0),
        #         self._franka_effort_limits[:7].unsqueeze(0)
        #     )
        
        # self._arm_control[:, :] = u_arm
        # import ipdb; ipdb.set_trace()
        # # Control gripper (binary open/close)
        # u_fingers = torch.zeros_like(self._gripper_control)
        # u_fingers[:, 0] = torch.where(
        #     u_gripper >= 0.0,
        #     self.franka_dof_upper_limits[-2].item(),
        #     self.franka_dof_lower_limits[-2].item()
        # )
        # u_fingers[:, 1] = torch.where(
        #     u_gripper >= 0.0,
        #     self.franka_dof_upper_limits[-1].item(),
        #     self.franka_dof_lower_limits[-1].item()
        # )
        # self._gripper_control[:, :] = u_fingers
        
        # # Deploy actions
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # print(u_arm, u_gripper)
        # print(self.cmd_limit, self.action_scale)

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.cfg.control.action_scale
        if self.cfg.control.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm

        # Control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
                                      self.franka_dof_lower_limits[-2].item())
        u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
                                      self.franka_dof_lower_limits[-1].item())
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        """Process state after physics simulation."""
        self.episode_length_buf += 1
        self.common_step_counter += 1
        
        # Refresh state tensors
        
        # ipdb.set_trace()
        # Check termination and compute rewards
        self.check_termination()
        self.compute_reward()
        
        # Reset environments that need it
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        
        # Compute observations
        self.compute_observations()
        
        # Store last actions
        self.last_actions[:] = self.actions[:]
        
        # Debug visualization
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """
        Check if environments need to be reset.
        
        Reset conditions:
        1. Episode timeout
        2. Cube fell off table
        """
        # Episode timeout
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_buf = self.time_out_buf.clone()
        
        # Cube fell off table (significant failure)
        table_height = self._table_surface_pos[2]
        cubeA_fell = self.states["cubeA_pos"][:, 2] < (table_height - 0.1)
        cubeB_fell = self.states["cubeB_pos"][:, 2] < (table_height - 0.1)
        self.reset_buf = self.reset_buf | cubeA_fell | cubeB_fell

    def compute_reward(self):
        """
        Compute rewards from registered reward functions.
        
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward.
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def compute_observations(self):
        """
        Compute observations for cube stacking task.
        
        Observations (19 dim for OSC, 26 for joint_tor):
            - cubeA_quat (4): Cube A orientation
            - cubeA_pos (3): Cube A position  
            - cubeA_to_cubeB_pos (3): Vector from cube A to cube B
            - eef_pos (3): End-effector position
            - eef_quat (4): End-effector orientation
            - q_gripper (2) for OSC mode, or q (9) for joint_tor mode
        """
        self._refresh()
        obs = [
            "cubeA_quat",
            "cubeA_pos", 
            "cubeA_to_cubeB_pos",
            "eef_pos",
            "eef_quat",
        ]
        
        # Add gripper state (2 dim) for OSC or full joint state (9 dim) for joint_tor
        if self.cfg.control.control_type == "osc":
            obs.append("q_gripper")
        else:
            obs.append("q")
        
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        
        # Debug: Check for NaN/Inf in observations
        if torch.any(~torch.isfinite(self.obs_buf)):
            print("NaN/Inf detected in observations!")
            print("Checking individual state components:")
            for ob_name in obs:
                val = self.states[ob_name]
                if torch.any(~torch.isfinite(val)):
                    print(f"  {ob_name}: contains NaN/Inf! shape={val.shape}")
                    print(f"    sample values: {val[0]}")
                else:
                    print(f"  {ob_name}: OK")
            # ipdb.set_trace()

    # -------- Reset Functions --------

    def _reset_cubes(self, env_ids: torch.Tensor):
        """
        Reset cube positions with collision-free sampling.
        
        Cubes are placed on the table with random XY positions and Z rotations.
        The sampling ensures cubes don't overlap with each other.
        
        Args:
            env_ids: Indices of environments to reset
        """
        if len(env_ids) == 0:
            return
        
        num_resets = len(env_ids)
        
        # Sample cube positions
        if self.cfg.cubes.fixed_positions:
            # Use fixed positions for debugging
            cubeA_pos = torch.tensor(
                [self._table_surface_pos[0] + self.cfg.cubes.cubeA_fixed_pos[0],
                 self._table_surface_pos[1] + self.cfg.cubes.cubeA_fixed_pos[1],
                 self._table_surface_pos[2] + self.cubeA_size / 2.0],
                device=self.device
            ).unsqueeze(0).repeat(num_resets, 1)
            
            cubeB_pos = torch.tensor(
                [self._table_surface_pos[0] + self.cfg.cubes.cubeB_fixed_pos[0],
                 self._table_surface_pos[1] + self.cfg.cubes.cubeB_fixed_pos[1],
                 self._table_surface_pos[2] + self.cubeB_size / 2.0],
                device=self.device
            ).unsqueeze(0).repeat(num_resets, 1)
        else:
            # Random positions with collision avoidance
            noise = self.cfg.cubes.start_position_noise
            
            # Sample cubeB first (it's the base, so position it more centrally)
            cubeB_xy = torch.zeros((num_resets, 2), device=self.device)
            cubeB_xy[:, 0] = self._table_surface_pos[0] + (2.0 * torch.rand(num_resets, device=self.device) - 1.0) * noise * 0.5
            cubeB_xy[:, 1] = self._table_surface_pos[1] + (2.0 * torch.rand(num_resets, device=self.device) - 1.0) * noise * 0.5
            
            cubeB_pos = torch.zeros((num_resets, 3), device=self.device)
            cubeB_pos[:, 0] = cubeB_xy[:, 0]
            cubeB_pos[:, 1] = cubeB_xy[:, 1]
            cubeB_pos[:, 2] = self._table_surface_pos[2] + self.cubeB_size / 2.0
            
            # Sample cubeA with minimum distance from cubeB
            min_dist = (self.cubeA_size + self.cubeB_size) / 2.0 + 0.03  # 3cm margin
            
            # Keep sampling until we have valid positions (not overlapping)
            valid_mask = torch.zeros(num_resets, dtype=torch.bool, device=self.device)
            cubeA_pos = torch.zeros((num_resets, 3), device=self.device)
            cubeA_pos[:, 2] = self._table_surface_pos[2] + self.cubeA_size / 2.0
            
            max_attempts = 100
            for _ in range(max_attempts):
                # Sample new positions for invalid cubes
                invalid_mask = ~valid_mask
                if not invalid_mask.any():
                    break
                
                num_invalid = invalid_mask.sum().item()
                cubeA_pos[invalid_mask, 0] = self._table_surface_pos[0] + (2.0 * torch.rand(num_invalid, device=self.device) - 1.0) * noise
                cubeA_pos[invalid_mask, 1] = self._table_surface_pos[1] + (2.0 * torch.rand(num_invalid, device=self.device) - 1.0) * noise
                
                # Check distance from cubeB
                dist_AB = torch.norm(cubeA_pos[:, :2] - cubeB_pos[:, :2], dim=-1)
                valid_mask = dist_AB >= min_dist
        
        # Sample random rotations around Z axis
        rot_noise = self.cfg.cubes.start_rotation_noise
        
        # CubeA rotation
        cubeA_rot_angles = 2.0 * rot_noise * torch.rand(num_resets, device=self.device) - rot_noise
        cubeA_axis_angle = torch.zeros((num_resets, 3), device=self.device)
        cubeA_axis_angle[:, 2] = cubeA_rot_angles
        cubeA_quat = axisangle2quat(cubeA_axis_angle)
        
        # CubeB rotation
        cubeB_rot_angles = 2.0 * rot_noise * torch.rand(num_resets, device=self.device) - rot_noise
        cubeB_axis_angle = torch.zeros((num_resets, 3), device=self.device)
        cubeB_axis_angle[:, 2] = cubeB_rot_angles
        cubeB_quat = axisangle2quat(cubeB_axis_angle)
        
        # Set cube states
        # Position (3) + Quaternion (4) + Linear velocity (3) + Angular velocity (3) = 13
        self._cubeA_state[env_ids, :3] = cubeA_pos
        self._cubeA_state[env_ids, 3:7] = cubeA_quat
        self._cubeA_state[env_ids, 7:] = 0.0  # Zero velocities
        
        self._cubeB_state[env_ids, :3] = cubeB_pos
        self._cubeB_state[env_ids, 3:7] = cubeB_quat
        self._cubeB_state[env_ids, 7:] = 0.0  # Zero velocities

    def reset_idx(self, env_ids: torch.Tensor):
        """
        Reset specified environments.
        
        Resets Franka arm pose and cube positions.
        
        Args:
            env_ids: Indices of environments to reset
        """
        if len(env_ids) == 0:
            return
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        
        # Reset Franka to default pose with optional noise
        reset_noise = torch.rand((len(env_ids), 9), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + 
            self.cfg.domain_rand.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0),
            self.franka_dof_upper_limits
        )
        
        # Gripper always resets to default (no noise)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]
        
        # Reset DOF states
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])
        
        # Reset control tensors
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)
        
        # Reset cubes
        self._reset_cubes(env_ids)
        
        # Deploy updates via indexed tensors
        # Need to update: franka (0), cubeA (3), cubeB (4)
        multi_env_ids = self._global_indices[env_ids, :].flatten()
        
        # Update root states (for cubes)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state.view(-1, 13)),
            gymtorch.unwrap_tensor(multi_env_ids),
            len(multi_env_ids)
        )
        
        # Update DOF states (for Franka)
        franka_indices = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(franka_indices),
            len(franka_indices)
        )
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._effort_control),
            gymtorch.unwrap_tensor(franka_indices),
            len(franka_indices)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(franka_indices),
            len(franka_indices)
        )
        
        # Reset episode tracking
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        
        # Reset episode sums for logging
        for key in self.episode_sums.keys():
            self.episode_sums[key][env_ids] = 0.0
        
        # Fill extras with episode info
        self._fill_extras(env_ids)

    def _fill_extras(self, env_ids: torch.Tensor):
        """Fill extras dict with episode information for logging."""
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
        
        # Send timeout info to algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _draw_debug_vis(self):
        """Draw debug visualizations for cube positions and targets."""
        self.gym.clear_lines(self.viewer)
        
        for i in range(min(self.num_envs, 8)):  # Only draw for first 8 envs
            env = self.envs[i]
            
            # Draw line from EEF to cubeA
            eef_pos = self.states["eef_pos"][i].cpu().numpy()
            cubeA_pos = self.states["cubeA_pos"][i].cpu().numpy()
            cubeB_pos = self.states["cubeB_pos"][i].cpu().numpy()
            
            # EEF to cubeA (red)
            self.gym.add_lines(self.viewer, env, 1,
                [eef_pos[0], eef_pos[1], eef_pos[2],
                 cubeA_pos[0], cubeA_pos[1], cubeA_pos[2]],
                [1.0, 0.0, 0.0])
            
            # CubeA to cubeB target position (green)
            target_pos = cubeB_pos.copy()
            target_pos[2] += self.cubeB_size / 2.0 + self.cubeA_size / 2.0
            self.gym.add_lines(self.viewer, env, 1,
                [cubeA_pos[0], cubeA_pos[1], cubeA_pos[2],
                 target_pos[0], target_pos[1], target_pos[2]],
                [0.0, 1.0, 0.0])

    # -------- Properties --------
    
    @property
    def table_surface_pos(self) -> np.ndarray:
        """Get the position of the table surface."""
        return self._table_surface_pos

    # -------- Reward Functions --------
    
    def _reward_dist(self) -> torch.Tensor:
        """Distance reward: encourages gripper to approach cubeA."""
        d_lf = torch.norm(self.states["cubeA_pos"] - self.states["eef_lf_pos"], dim=-1)
        d_rf = torch.norm(self.states["cubeA_pos"] - self.states["eef_rf_pos"], dim=-1)
        return 1.0 - torch.tanh(10.0 * (d_lf + d_rf) / 2.0)
    
    def _reward_lift(self) -> torch.Tensor:
        """Lift reward: bonus for lifting cubeA above table."""
        cubeA_lifted = self.states["cubeA_pos"][:, 2] > (self._cubeA_initial_height + self.lift_height_threshold)
        return cubeA_lifted.float()
    
    def _reward_align(self) -> torch.Tensor:
        """Alignment reward: encourages cubeA to be aligned above cubeB."""
        # Target position
        target_pos = self.states["cubeB_pos"].clone()
        target_pos[:, 2] += self.cubeB_size / 2.0 + self.cubeA_size / 2.0
        
        cubeA_to_target = torch.norm(self.states["cubeA_pos"] - target_pos, dim=-1)
        cubeA_lifted = self.states["cubeA_pos"][:, 2] > (self._cubeA_initial_height + self.lift_height_threshold)
        
        return (1.0 - torch.tanh(10.0 * cubeA_to_target)) * cubeA_lifted.float()
    
    def _reward_stack(self) -> torch.Tensor:
        """Stack reward: large bonus for successful stacking."""
        target_pos = self.states["cubeB_pos"].clone()
        target_pos[:, 2] += self.cubeB_size / 2.0 + self.cubeA_size / 2.0
        
        # Check alignment
        cubeA_above_cubeB_xy = torch.norm(
            self.states["cubeA_pos"][:, :2] - self.states["cubeB_pos"][:, :2], 
            dim=-1
        ) < self.stack_xy_threshold
        cubeA_on_cubeB_z = torch.abs(
            self.states["cubeA_pos"][:, 2] - target_pos[:, 2]
        ) < self.stack_z_threshold
        
        # Check if gripper released
        d_lf = torch.norm(self.states["cubeA_pos"] - self.states["eef_lf_pos"], dim=-1)
        d_rf = torch.norm(self.states["cubeA_pos"] - self.states["eef_rf_pos"], dim=-1)
        gripper_away = (d_lf > self.gripper_away_threshold) & (d_rf > self.gripper_away_threshold)
        
        stacked = cubeA_above_cubeB_xy & cubeA_on_cubeB_z & gripper_away
        return stacked.float()
