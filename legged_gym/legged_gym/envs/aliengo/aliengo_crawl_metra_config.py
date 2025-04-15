import numpy as np
from legged_gym.envs.aliengo.aliengo_crawl_config import AliengoCrawlCfg, AliengoCrawlCfgPPO
from legged_gym.utils.helpers import merge_dict


class AliengoCrawlMetraCfg(AliengoCrawlCfg):
    #### uncomment this to train non-virtual terrain
    # class sensor( AliengoFieldCfg.sensor ):
    #     class proprioception( AliengoFieldCfg.sensor.proprioception ):
    #         delay_action_obs = True
    #         latency_range = [0.04-0.0025, 0.04+0.0075]
    #### uncomment the above to train non-virtual terrain

    class env(AliengoCrawlCfg.env):
        skill_dim = 1
        # phi_start_dim = 2
        # phi_input_dim = 4
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
        episode_length_s=20
    class terrain( AliengoCrawlCfg.terrain ):
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.
        curriculum = False

        BarrierTrack_kwargs = merge_dict(AliengoCrawlCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "crawl",
            ],
            track_block_length= 1.6,
            crawl= dict(
                height= (0.50, 0.50),
                depth= (0.1, 0.6), # size along the forward axis
                wall_height= 0.6,
                no_perlin_at_obstacle= False,
            ),
            virtual_terrain= False, # Change this to False for real terrain
        ))

        TerrainPerlin_kwargs = merge_dict(AliengoCrawlCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= 0.1,
        ))

    class commands(AliengoCrawlCfg.commands):
        class ranges(AliengoCrawlCfg.commands.ranges):
            lin_vel_x = [0.3, 0.8]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]

    class termination(AliengoCrawlCfg.termination):
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            "out_of_track",
        ]

    class domain_rand(AliengoCrawlCfg.domain_rand):
        # push_robots = True # use for virtual training
        push_robots = False  # use for non-virtual training

    class rewards(AliengoCrawlCfg.rewards):
        class scales:
            tracking_ang_vel = 0.05
            world_vel_l2norm = -1.0
            legs_energy_substeps = -1e-5
            alive = 2.0
            penetrate_depth = 0.#-3e-3
            penetrate_volume = 0.#-3e-3
            exceed_dof_pos_limits = -1e-2#-1e-2
            exceed_torque_limits_i = -2e-1#-2e-1
            diversity = 0.1

        only_positive_rewards = False # if true ne
    
    class normalization(AliengoCrawlCfg.normalization):
        class obs_scales(AliengoCrawlCfg.normalization.obs_scales):
            base_pose = [1., 1., 1., 1., 1., 1.]
    class noise(AliengoCrawlCfg.noise):
        add_noise=True

    class curriculum(AliengoCrawlCfg.curriculum):
        penetrate_volume_threshold_harder = 4000
        penetrate_volume_threshold_easier = 10000
        penetrate_depth_threshold_harder = 100
        penetrate_depth_threshold_easier = 300


class AliengoCrawlMetraCfgPPO(AliengoCrawlCfgPPO):
    seed=5
    class algorithm(AliengoCrawlCfgPPO.algorithm):
        add_skill_discovery_loss = True
        add_next_state = True
        adjustable_kappa = True
        kappa_cap = 10.0
        kappa = 10
    
        
    class runner(AliengoCrawlCfgPPO.runner):
        policy_class_name = 'ActorCriticMetra'
        experiment_name = 'aliengo_crawl_metra'
        algorithm_class_name = 'PPOMetra'
        max_iterations = 15000  # number of policy updates
        save_interval = 1000
        resume = False

    class policy(AliengoCrawlCfgPPO.policy):
        # for z
        phi_start_dim = 2
        phi_input_dim = 1
        phi_index = [2]
        skill_dim = 1
