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

from legged_gym.envs.base.base_config import BaseConfig


class FrankaCubeStackCfg(BaseConfig):
    """Configuration for Franka cube stacking task.
    
    The task involves a Franka arm picking up cubeA and stacking it on top of cubeB.
    
    Observations (19 dim for OSC, 26 for joint_tor):
        - cubeA_quat (4): Cube A orientation quaternion (x, y, z, w)
        - cubeA_pos (3): Cube A position (x, y, z)
        - cubeA_to_cubeB_pos (3): Vector from cube A to cube B
        - eef_pos (3): End-effector position
        - eef_quat (4): End-effector orientation quaternion
        - q_gripper (2) for OSC mode, q (9) for joint_tor mode
    
    Actions (7 dim for OSC, 8 for joint_tor):
        - OSC mode: delta end-effector pose (6) + gripper command (1)
        - joint_tor mode: joint torques (7) + gripper command (1)
    """
    
    class env:
        num_envs = 8192
        num_observations = 19  # OSC mode: cubeA_pose(7) + cubeB_pos(3) + eef_pose(7) + q_gripper(2)
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step()
        num_actions = 7  # OSC mode: delta EEF (6) + gripper (1)
        env_spacing = 1.5  # spacing between environments
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 5.0  # episode length in seconds (300 steps at 60Hz)
        aggregate_mode = 3  # aggregate mode for performance optimization
        
        # Debug visualization
        enable_debug_vis = False

    class control:
        control_type = 'osc'  # 'osc' (Operational Space Control) or 'joint_tor' (joint torques)
        action_scale = 1.0
        # OSC gains
        kp = [150.0] * 6  # Position gain for 6-DOF end-effector control
        kd_factor = 2.0  # Damping: kd = kd_factor * sqrt(kp)
        # Nullspace control gains (prevents large joint configuration changes)
        kp_null = [10.0] * 7  # Nullspace position gain for 7 arm joints
        kd_null_factor = 2.0  # Nullspace damping factor
        # Command limits for OSC mode [x, y, z, roll, pitch, yaw]
        osc_cmd_limit = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 1

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/franka/robots/franka_panda_gripper.urdf"
        name = "franka"  # actor name
        disable_gravity = True  # Franka arm mounted, disable gravity
        collapse_fixed_joints = False  # keep all joints for accurate control
        fix_base_link = True  # fix the base of the robot
        default_dof_drive_mode = 3  # 0: none, 1: pos tgt, 2: vel tgt, 3: effort
        self_collisions = 0  # 1 to disable, 0 to enable
        flip_visual_attachments = True  # required for Franka meshes
        use_mesh_materials = True
        
        # Physics properties
        thickness = 0.001
        # angular_damping = 0.0
        # linear_damping = 0.0
        # max_angular_velocity = 1000.0
        # max_linear_velocity = 1000.0
        # armature = 0.0
        
        # DOF stiffness and damping (7 arm joints + 2 gripper joints)
        # Arm joints (0-6): effort control with zero stiffness
        # Gripper joints (7-8): position control with high stiffness
        dof_stiffness = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5000.0, 5000.0]
        dof_damping = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0]
        
        # Gripper effort limits (will be overridden)
        gripper_effort_limit = 200.0
        gripper_speed_scale = 0.1

    class init_state:
        # Default Franka arm joint positions (7 arm + 2 gripper)
        # These are the canonical "ready" pose positions
        default_joint_angles = {
            "panda_joint1": 0.0,
            "panda_joint2": 0.1963,
            "panda_joint3": 0.0,
            "panda_joint4": -2.6180,
            "panda_joint5": 0.0,
            "panda_joint6": 2.9416,
            "panda_joint7": 0.7854,
            "panda_finger_joint1": 0.035,
            "panda_finger_joint2": 0.035,
        }
        # Franka start pose (relative to table)
        pos = [-0.45, 0.0, 0.0]  # x, y, z offset from table stand top
        rot = [0.0, 0.0, 0.0, 1.0]  # x, y, z, w quaternion

    class table:
        # Table dimensions and position
        pos = [0.0, 0.0, 1.0]  # x, y, z [m]
        size = [1.2, 1.2, 0.05]  # length, width, thickness [m]
        # Table stand (under Franka base)
        stand_height = 0.1
        stand_size = [0.2, 0.2]  # length, width [m]
        stand_offset = [-0.5, 0.0]  # x, y offset from table center

    class cubes:
        """Cube configuration parameters."""
        # Cube sizes in meters (side length)
        cubeA_size = 0.050  # Small cube (to be picked and stacked)
        cubeB_size = 0.070  # Large cube (base for stacking)
        
        # Cube colors (RGB values 0-1)
        cubeA_color = [0.6, 0.1, 0.0]  # Orange/red
        cubeB_color = [0.0, 0.4, 0.1]  # Green
        
        # Cube spawn randomization
        # start_position_noise: cubes are sampled within ±noise around table center
        start_position_noise = 0.25  # meters
        # start_rotation_noise: random rotation around z-axis in radians
        start_rotation_noise = 0.785  # ~45 degrees
        
        # Fixed cube positions (optional, for debugging/testing)
        fixed_positions = False
        cubeA_fixed_pos = [-0.1, 0.0]  # x, y offset from table center
        cubeB_fixed_pos = [0.1, 0.0]   # x, y offset from table center

    class domain_rand:
        randomize_friction = False
        friction_range = [0.5, 1.25]
        # Franka position/rotation randomization
        franka_position_noise = 0.0
        franka_rotation_noise = 0.0
        franka_dof_noise = 0.25  # noise added to default DOF positions at reset

    class rewards:
        """Reward configuration for cube stacking task."""
        
        class scales:
            # Distance reward: encourages gripper to approach cubeA
            # reward = scale * (1 - tanh(10 * avg_distance))
            dist = 0.1
            
            # Lift reward: bonus for lifting cubeA above table
            # reward = scale * (cubeA_lifted) where cubeA_lifted is binary
            lift = 1.5
            
            # Alignment reward: encourages cubeA to be aligned above cubeB
            # reward = scale * (1 - tanh(10 * distance_to_target)) * cubeA_lifted
            align = 2.0
            
            # Stack reward: large bonus for successful stacking
            # Awarded when cubeA is on cubeB and gripper has released
            stack = 16.0
        
        # Whether to clip negative rewards
        only_positive_rewards = False
        
        # Success thresholds
        lift_height_threshold = 0.04  # cubeA must be lifted this much above initial height
        stack_xy_threshold = 0.02  # horizontal alignment tolerance for stacking
        stack_z_threshold = 0.02  # vertical position tolerance for stacking
        gripper_away_threshold = 0.04  # gripper must be this far from cubeA for stack success

    class normalization:
        clip_observations = 5.0
        clip_actions = 1.0

    class noise:
        add_noise = False  # disable noise for manipulation tasks by default
        noise_level = 1.0

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [2.0, 2.0, 2.5]  # [m] - positioned to view table
        lookat = [0.0, 0.0, 1.0]  # [m] - looking at table center

    class sim:
        dt = 0.01667  # 1/60 Hz (60 Hz simulation)
        substeps = 2
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z
        no_camera = True  # if True, disable rendering when headless

        class physx:
            num_threads = 4
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8
            num_velocity_iterations = 1
            contact_offset = 0.005  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.2  # [m/s]
            max_depenetration_velocity = 1000.0
            max_gpu_contact_pairs = 2**20  # 1024*1024
            default_buffer_size_multiplier = 5.0
            contact_collection = 0  # 0: never, 1: last sub-step, 2: all sub-steps
            num_subscenes = 4

class FrankaCubeStackCfgPPO(BaseConfig):
    """PPO configuration for Franka cube stacking task."""
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
    class algorithm:
        # Training params
        value_loss_coef = 4.0  # Higher value loss coefficient for manipulation
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0  # No entropy bonus for manipulation
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs * nsteps / nminibatches
        learning_rate = 5.e-4
        schedule = 'adaptive'  # adaptive learning rate
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.008
        max_grad_norm = 1.0
        clip_min_std = 1e-15
        add_skill_discovery_loss = False  # disable skill discovery for this task
        
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 32  # per iteration (horizon length)
        max_iterations = 10000  # number of policy updates
        
        # Logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'franka_cube_stack'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
