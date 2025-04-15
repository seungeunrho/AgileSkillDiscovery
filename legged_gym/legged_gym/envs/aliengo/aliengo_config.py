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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class AliengoRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        use_lin_vel = True
        num_observations = 235 # no measure_heights makes num_obs = 48; with measure_heights makes num_obs 235
    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #     'FR_hip_joint': -0.1 ,  # [rad]
        #     'FL_hip_joint': 0.1,   # [rad]
        #     'RR_hip_joint': -0.1,   # [rad]
        #     'RL_hip_joint': 0.1,   # [rad]

        #     'FL_thigh_joint': 0.8,     # [rad]
        #     'RL_thigh_joint': 1.,   # [rad]
        #     'FR_thigh_joint': 0.8,     # [rad]
        #     'RR_thigh_joint': 1.,   # [rad]

        #     'FL_calf_joint': -1.5,   # [rad]
        #     'RL_calf_joint': -1.5,    # [rad]
        #     'FR_calf_joint': -1.5,  # [rad]
        #     'RR_calf_joint': -1.5,    # [rad]
        # }
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FR_hip_joint': -0.0 ,  # [rad]
            'FL_hip_joint': 0.0,   # [rad]
            'RR_hip_joint': -0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.6,     # [rad]
            'RL_thigh_joint': 0.6,   # [rad]
            'FR_thigh_joint': 0.6,     # [rad]
            'RR_thigh_joint': 0.6,   # [rad]

            'FL_calf_joint': -1.25,   # [rad]
            'RL_calf_joint': -1.25,    # [rad]
            'FR_calf_joint': -1.25,  # [rad]
            'RR_calf_joint': -1.25,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo/urdf/aliengo.urdf'
        name = "aliengo"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        sdk_dof_range = dict( # copied from aliengo_const.h from unitree_legged_sdk
            Hip_max= 1.2217304764,
            Hip_min=-1.2217304764,
            Thigh_max= 4.18879020479,
            Thigh_min= -1.0471975512,
            Calf_max= -0.645771823238,
            Calf_min= -2.77507351067,
        )
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.5
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0

class AliengoRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'full'
        experiment_name = 'rough_aliengo'

#### To train the model with partial observation ####

class AliengoPlaneCfg( AliengoRoughCfg ):
    class env( AliengoRoughCfg.env ):
        use_lin_vel = False
        num_observations = 48

    class control( AliengoRoughCfg.control ):
        stiffness = {'joint': 40.}

    class domain_rand( AliengoRoughCfg.domain_rand ):
        randomize_base_mass = True

    class terrain( AliengoRoughCfg.terrain ):
        mesh_type = "trimesh"
        measure_heights = True

class AliengoRoughCfgTPPO( AliengoRoughCfgPPO ):

    class algorithm( AliengoRoughCfgPPO.algorithm ):
        distillation_loss_coef = 50.

        teacher_ac_path = "logs/rough_aliengo/Nov08_07-55-33_full/model_1500.pt"
        teacher_policy_class_name = AliengoRoughCfgPPO.runner.policy_class_name
        class teacher_policy( AliengoRoughCfgPPO.policy ):
            num_actor_obs = 235
            num_critic_obs = 235
            num_actions = 12
    
    class runner( AliengoRoughCfgPPO.runner ):
        algorithm_class_name = "TPPO"
        run_name = 'nolinvel_plane_Kp25_aclip1.5_privilegedHeights_distillation50_randomizeMass'
        experiment_name = 'teacher_aliengo'
    
  