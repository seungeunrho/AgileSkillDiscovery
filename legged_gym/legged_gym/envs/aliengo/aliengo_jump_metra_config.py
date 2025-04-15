from os import path as osp
import numpy as np
from legged_gym.envs.aliengo.aliengo_jump_config import AliengoJumpCfg, AliengoJumpCfgPPO
from legged_gym.utils.helpers import merge_dict

class AliengoJumpMetraCfg( AliengoJumpCfg ):

    # class init_state( AliengoJumpCfg.init_state ):
    #     pos = [0., 0., 0.45]

    #### uncomment this to train non-virtual terrain
    # class sensor( AliengoJumpCfg.sensor ):
    #     class proprioception( AliengoJumpCfg.sensor.proprioception ):
    #         latency_range = [0.04-0.0025, 0.04+0.0075]
    #### uncomment the above to train non-virtual terrain
    class env(AliengoJumpCfg.env):
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
    class terrain( AliengoJumpCfg.terrain ):
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.
        curriculum = False

        BarrierTrack_kwargs = merge_dict(AliengoJumpCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "jump",
            ],
            track_block_length= 1.6,
            jump= dict(
                height= (0.40, 0.40), # use this to train in virtual terrain
                # height= (0.1, 0.5), # use this to train in non-virtual terrain
                depth= (0.1, 0.2),
                fake_offset= 0.0, # [m] an offset that make the robot easier to get into the obstacle
                jump_down_prob= 0., # probability of jumping down use it in non-virtual terrain
            ),
            virtual_terrain= False, # Change this to False for real terrain
            no_perlin_threshold= 0.12,
            n_obstacles_per_track= 3,
        ))

        TerrainPerlin_kwargs = merge_dict(AliengoJumpCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= [0.05, 0.15],
        # TerrainPerlin_kwargs = merge_dict(AliengoJumpCfg.terrain.TerrainPerlin_kwargs, dict(
        #     zScale= [0.0, 0.0],
        ))
    
    class commands( AliengoJumpCfg.commands ):
        class ranges( AliengoJumpCfg.commands.ranges ):
            lin_vel_x = [0.8, 1.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]

    class termination( AliengoJumpCfg.termination ):
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            "out_of_track",
        ]
    class control( AliengoJumpCfg.control ):
        stiffness = {'joint': 70.}
        damping = {'joint': 1.5}
        action_scale = 0.5
        torque_limits = 40 # override the urdf
        computer_clip_torque = True
        motor_clip_torque = True
       
    class domain_rand( AliengoJumpCfg.domain_rand ):
        push_robots = False

    class rewards( AliengoJumpCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.1
            world_vel_l2norm = -1.0
            legs_energy_substeps = -1e-6
            alive = 2.
            penetrate_depth = 0 #-1e-2
            penetrate_volume = 0. #-1e-2
            exceed_dof_pos_limits = -1e-1
            exceed_torque_limits_i = -2e-1
            diversity=0.1
        soft_dof_pos_limit = 0.8
        max_contact_force = 100.0

class AliengoJumpMetraCfgPPO( AliengoJumpCfgPPO ):
    seed=1
    class algorithm( AliengoJumpCfgPPO.algorithm ):
        add_skill_discovery_loss = True
        add_next_state = True
        adjustable_kappa = True
        kappa_cap = 15
        kappa = 15
    
    class runner( AliengoJumpCfgPPO.runner ):
        policy_class_name = 'ActorCriticMetra'
        experiment_name = 'aliengo_jump_metra'
        algorithm_class_name = 'PPOMetra'
        max_iterations = 25000  # number of policy updates
        save_interval = 1000
        resume = False
    
    class policy(AliengoJumpCfgPPO.policy):
        # for xyz
        # phi_start_dim = 0
        # phi_input_dim = 3
        # skill_dim = 2
        
          # for x
        # phi_start_dim = 0
        # phi_input_dim = 1
        # skill_dim = 2
     # for v_x and v_z
        phi_start_dim = 2
        phi_input_dim = 1
        phi_index = [2]
        skill_dim = 1