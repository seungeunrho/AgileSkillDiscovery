import numpy as np
from legged_gym.envs.aliengo.aliengo_leap_config import AliengoLeapCfg, AliengoLeapCfgPPO
from legged_gym.utils.helpers import merge_dict

class AliengoLeapMetraCfg( AliengoLeapCfg ):

    #### uncomment this to train non-virtual terrain
    # class sensor( AliengoFieldCfg.sensor ):
    #     class proprioception( AliengoFieldCfg.sensor.proprioception ):
    #         delay_action_obs = True
    #         latency_range = [0.04-0.0025, 0.04+0.0075]
    #### uncomment the above to train non-virtual terrain

    class env(AliengoLeapCfg.env):
        skill_dim = 1
        # phi_start_dim = 2
        # phi_input_dim = 1
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
    class terrain( AliengoLeapCfg.terrain ):
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.
        curriculum = False

        BarrierTrack_kwargs = merge_dict(AliengoLeapCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "leap",
            ],
            leap= dict(
                length= (0.55, 0.55),
                depth= (0.4, 0.8),
                height= 0.2,
            ),
            virtual_terrain= False, # Change this to False for real terrain
            no_perlin_threshold= 0.06,
            # add_perlin_noise = False
        ))

        TerrainPerlin_kwargs = merge_dict(AliengoLeapCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= [0.05, 0.1],
        ))
        # TerrainPerlin_kwargs = merge_dict(AliengoLeapCfg.terrain.TerrainPerlin_kwargs, dict(
        #     zScale= [0.0, 0.0],
        # ))
    
    class commands( AliengoLeapCfg.commands ):
        class ranges( AliengoLeapCfg.commands.ranges ):
            lin_vel_x =  [1.0, 1.5] #[2.0,3.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]

    class termination( AliengoLeapCfg.termination ):
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            "out_of_track",
        ]
    class control( AliengoLeapCfg.control ):
        stiffness = {'joint': 50.}
        damping = {'joint': 1.0}
        action_scale = 0.5
        torque_limits = 40 # override the urdf
        computer_clip_torque = False
        motor_clip_torque = False
        
    class domain_rand(AliengoLeapCfg.domain_rand):
        # push_robots = True # use for virtual training
        push_robots = False  # use for non-virtual training

    class rewards( AliengoLeapCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.05
            world_vel_l2norm = -1.0 #-1.
            legs_energy_substeps = -1e-7 #-1e-6
            alive = 2 #2.
            # tracking_lin_vel = 1
            penetrate_depth = 0. #-4e-3
            penetrate_volume = 0. #-4e-3
            exceed_dof_pos_limits = -1e-2#-1e-1 #-1e-1
            exceed_torque_limits_i = -2e-2#-2e-1 #-2e-1
            diversity = 0.1
            
        only_positive_rewards = False
    
    class normalization(AliengoLeapCfg.normalization):
        class obs_scales(AliengoLeapCfg.normalization.obs_scales):
            base_pose = [1., 1., 1., 1., 1., 1.]

    class noise(AliengoLeapCfg.noise):
        add_noise=True

    class curriculum( AliengoLeapCfg.curriculum ):
        penetrate_volume_threshold_harder = 9000
        penetrate_volume_threshold_easier = 10000
        penetrate_depth_threshold_harder = 300
        penetrate_depth_threshold_easier = 5000


class AliengoLeapMetraCfgPPO( AliengoLeapCfgPPO ):
    seed=1
    class algorithm( AliengoLeapCfgPPO.algorithm ):
        add_skill_discovery_loss = True
        add_next_state = True
        adjustable_kappa = True
        kappa_cap = 10
        kappa = 10
    
    class runner( AliengoLeapCfgPPO.runner ):
        policy_class_name = 'ActorCriticMetra'
        experiment_name = 'aliengo_leap_metra'
        algorithm_class_name = 'PPOMetra'
        max_iterations = 10000  # number of policy updates
        save_interval = 1000
        resume = False
        # load_run = "/nethome/kgarg65/flash/Agile_oc/Agile_oc/AgileSkillDiscovery/legged_gym/logs/field_aliengo_metra/Jan27_22-45-36_time_20s_oc_seed3"
        # checkpoint = 5000
        # init_critic = True
    
    class policy(AliengoLeapCfgPPO.policy):
        # for xyz
        # phi_start_dim = 0
        # phi_input_dim = 3
        # skill_dim = 2
        
          # for x
        # phi_start_dim = 0
        # phi_input_dim = 1
        # skill_dim = 2
     # for v_x
        phi_start_dim = 6
        phi_input_dim = 1
        skill_dim = 1
        phi_index = [6]