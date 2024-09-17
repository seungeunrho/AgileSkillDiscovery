import numpy as np
import os.path as osp
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO, A1PlaneCfg
from legged_gym.envs.a1.a1_field_config import A1FieldCfg, A1FieldCfgPPO

class A1FieldMetraCfg(A1FieldCfg):
    class env(A1FieldCfg.env):
        num_envs = 4096
        skill_dim = 2
        phi_start_dim = 0
        phi_input_dim = 2
        sample_skill = True
        obs_components = [
            "base_pose",
            "proprioception",  # 48
            # "height_measurements", # 187
            "skills",
        ]
        
        
    class terrain(A1FieldCfg.terrain):
        TerrainPerlin_kwargs = dict(
            # zScale= [0.08, 0.15],
            zScale= 0.15, # Use a constant zScale for training a walk policy
            frequency= 10,
        )
    
    class domain_rand(A1FieldCfg.domain_rand):
        randomize_motor = False
        leg_motor_strength_range = [1.0, 1.0]
        randomize_base_mass = False
        added_mass_range = [2.0, 2.0]
        randomize_friction = False
        friction_range = [1., 1.]
    
    class rewards(A1FieldCfg.rewards):
        class scales:
            tracking_ang_vel = 0.0
            world_vel_l2norm = 0.0
            legs_energy_substeps = -2e-5
            legs_energy = -0.
            alive = 2.
            # penalty for hardware safety
            exceed_dof_pos_limits = -1e-1
            exceed_torque_limits_i = -2e-1
            diversity = 100.0
        soft_dof_pos_limit = 0.01
    
    class normalization(A1FieldCfg.normalization):
        class obs_scales(A1FieldCfg.normalization.obs_scales):
            base_pose = [1., 1., 1., 1., 1., 1.]

class A1FieldMetraCfgPPO(A1FieldCfgPPO):
    class algorithm(A1FieldCfgPPO.algorithm):
        add_skill_discovery_loss = True
        add_next_state = True
        adjustable_kappa = True
    
    class runner(A1FieldCfgPPO.runner):
        policy_class_name = "ActorCriticMetra"
        experiment_name = "field_a1_metra"
        algorithm_class_name = 'PPOMetra'
        max_iterations = 200000  # number of policy updates
        save_interval = 1000
        resume = False
    
    class policy(A1FieldCfgPPO.policy):
        phi_input_dim = 2
        skill_dim = 2
    