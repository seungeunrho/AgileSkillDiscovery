import numpy as np
from legged_gym.envs.a1.a1_tilt_config import A1TiltCfg, A1TiltCfgPPO
from legged_gym.utils.helpers import merge_dict


class A1TiltMetraCfg(A1TiltCfg):
    #### uncomment this to train non-virtual terrain
    # class sensor( A1FieldCfg.sensor ):
    #     class proprioception( A1FieldCfg.sensor.proprioception ):
    #         delay_action_obs = True
    #         latency_range = [0.04-0.0025, 0.04+0.0075]
    #### uncomment the above to train non-virtual terrain

    class env(A1TiltCfg.env):
        skill_dim = 2
        # phi_start_dim = 3
        # phi_input_dim = 3
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
        episode_length_s=5

    class terrain(A1TiltCfg.terrain):
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.
        curriculum = False

        BarrierTrack_kwargs = merge_dict(A1TiltCfg.terrain.BarrierTrack_kwargs, dict(
            options=[
                "tilt",
            ],
            tilt=dict(
                width=(0.35, 0.35),
                depth=(0.4, 1.),  # size along the forward axis
                opening_angle=0.0,  # [rad] an opening that make the robot easier to get into the obstacle
                wall_height=0.5,
            ),
            virtual_terrain=False,  # Change this to False for real terrain
            no_perlin_threshold=0.06,
        ))

        TerrainPerlin_kwargs = merge_dict(A1TiltCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale=[0.05, 0.1],
        ))

    class commands(A1TiltCfg.commands):
        class ranges(A1TiltCfg.commands.ranges):
            lin_vel_x = [0.3, 0.6]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]

    class termination(A1TiltCfg.termination):
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            "out_of_track",
        ]

    class domain_rand(A1TiltCfg.domain_rand):
        # push_robots = True # use for virtual training
        push_robots = False  # use for non-virtual training

    class noise(A1TiltCfg.noise):
        add_noise=False
    class rewards(A1TiltCfg.rewards):
        class scales:
            tracking_ang_vel = 0.05
            world_vel_l2norm = -1.
            legs_energy_substeps = -1e-5
            alive = 2.
            penetrate_depth = 0.#-3e-3
            penetrate_volume = 0.#-3e-3
            exceed_dof_pos_limits = -1e-1
            exceed_torque_limits_i = -2e-1
            diversity = 0.1

        only_positive_rewards = False # if true ne
    
    class normalization(A1TiltCfg.normalization):
        class obs_scales(A1TiltCfg.normalization.obs_scales):
            base_pose = [1., 1., 1., 1., 1., 1.]

    class curriculum(A1TiltCfg.curriculum):
        penetrate_volume_threshold_harder = 4000
        penetrate_volume_threshold_easier = 10000
        penetrate_depth_threshold_harder = 100
        penetrate_depth_threshold_easier = 300


class A1TiltMetraCfgPPO(A1TiltCfgPPO):
    class algorithm(A1TiltCfgPPO.algorithm):
        add_skill_discovery_loss = True
        add_next_state = True
        adjustable_kappa = True

    class runner(A1TiltCfgPPO.runner):
        policy_class_name = 'ActorCriticMetra'
        experiment_name = 'a1_tilt_metra'
        algorithm_class_name = 'PPOMetra'
        max_iterations = 200000  # number of policy updates
        save_interval = 1000
        resume = False

    class policy(A1TiltCfgPPO.policy):
        # for x y z
        # phi_start_dim = 0
        # phi_input_dim = 3
        
        # for r
        phi_start_dim = 0
        phi_input_dim = 4
        skill_dim = 2
