import numpy as np
from legged_gym.envs.a1.a1_leap_config import A1LeapCfg, A1LeapCfgPPO
from legged_gym.utils.helpers import merge_dict

class A1LeapMetraCfg( A1LeapCfg ):

    #### uncomment this to train non-virtual terrain
    # class sensor( A1FieldCfg.sensor ):
    #     class proprioception( A1FieldCfg.sensor.proprioception ):
    #         delay_action_obs = True
    #         latency_range = [0.04-0.0025, 0.04+0.0075]
    #### uncomment the above to train non-virtual terrain

    class env(A1LeapCfg.env):
        skill_dim = 2
        phi_start_dim = 0
        phi_input_dim = 6
        sample_skill = True
        obs_components = [
            "base_pose",
            "proprioception",  # 48
            # "height_measurements", # 187

            "robot_config",
            "engaging_block",
            "sidewall_distance",
            "skills",
        ]
    
    class terrain( A1LeapCfg.terrain ):
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.
        curriculum = False

        BarrierTrack_kwargs = merge_dict(A1LeapCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "leap",
            ],
            leap= dict(
                length= (1.0, 1.0),
                depth= (0.4, 0.8),
                height= 0.2,
            ),
            virtual_terrain= False, # Change this to False for real terrain
            no_perlin_threshold= 0.06,
        ))

        TerrainPerlin_kwargs = merge_dict(A1LeapCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= [0.05, 0.1],
        ))
    
    class commands( A1LeapCfg.commands ):
        class ranges( A1LeapCfg.commands.ranges ):
            lin_vel_x = [1.0, 1.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]

    class termination( A1LeapCfg.termination ):
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            "out_of_track",
        ]
    
    class domain_rand(A1LeapCfg.domain_rand):
        # push_robots = True # use for virtual training
        push_robots = False  # use for non-virtual training

    class rewards( A1LeapCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.05
            world_vel_l2norm = -1.
            legs_energy_substeps = -1e-6
            alive = 2.
            penetrate_depth = 0. #-4e-3
            penetrate_volume = 0. #-4e-3
            exceed_dof_pos_limits = -1e-1
            exceed_torque_limits_i = -2e-1
            diversity = 100.0
        
        only_positive_rewards = False
    
    class normalization(A1LeapCfg.normalization):
        class obs_scales(A1LeapCfg.normalization.obs_scales):
            base_pose = [1., 1., 1., 1., 1., 1.]

    class curriculum( A1LeapCfg.curriculum ):
        penetrate_volume_threshold_harder = 9000
        penetrate_volume_threshold_easier = 10000
        penetrate_depth_threshold_harder = 300
        penetrate_depth_threshold_easier = 5000


class A1LeapMetraCfgPPO( A1LeapCfgPPO ):
    class algorithm( A1LeapCfgPPO.algorithm ):
        add_skill_discovery_loss = True
        add_next_state = True
    
    class runner( A1LeapCfgPPO.runner ):
        policy_class_name = 'ActorCriticMetra'
        experiment_name = 'a1_leap_metra'
        algorithm_class_name = 'PPOMetra'
        max_iterations = 200000  # number of policy updates
        save_interval = 1000
        resume = False
    
    class policy(A1LeapCfgPPO.policy):
        phi_input_dim = 6
        skill_dim = 2
    