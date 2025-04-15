from os import path as osp
import numpy as np
from legged_gym.envs.aliengo.aliengo_field_config import AliengoFieldCfg, AliengoFieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class AliengoJumpCfg( AliengoFieldCfg ):

    class init_state( AliengoFieldCfg.init_state ):
        pos = [0., 0., 0.55]

    #### uncomment this to train non-virtual terrain
    # class sensor( AliengoFieldCfg.sensor ):
    #     class proprioception( AliengoFieldCfg.sensor.proprioception ):
    #         latency_range = [0.04-0.0025, 0.04+0.0075]
    #### uncomment the above to train non-virtual terrain
    
    class terrain( AliengoFieldCfg.terrain ):
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.
        curriculum = True

        BarrierTrack_kwargs = merge_dict(AliengoFieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "jump",
            ],
            track_block_length= 1.6,
            jump= dict(
                height= (0.2, 0.6), # use this to train in virtual terrain
                # height= (0.1, 0.5), # use this to train in non-virtual terrain
                depth= (0.1, 0.2),
                fake_offset= 0.0, # [m] an offset that make the robot easier to get into the obstacle
                jump_down_prob= 0., # probability of jumping down use it in non-virtual terrain
            ),
            virtual_terrain= False, # Change this to False for real terrain
            no_perlin_threshold= 0.12,
            n_obstacles_per_track= 3,
        ))

        TerrainPerlin_kwargs = merge_dict(AliengoFieldCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= [0.05, 0.15],
        ))
    
    class commands( AliengoFieldCfg.commands ):
        class ranges( AliengoFieldCfg.commands.ranges ):
            lin_vel_x = [0.8, 1.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]

    class termination( AliengoFieldCfg.termination ):
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            "out_of_track",
        ]
        z_low_kwargs = merge_dict(AliengoFieldCfg.termination.z_low_kwargs, dict(
            threshold= -1.,
        ))

    class domain_rand( AliengoFieldCfg.domain_rand ):
        class com_range( AliengoFieldCfg.domain_rand.com_range ):
            z = [-0.1, 0.1]
        
        init_base_pos_range = dict(
            x= [0.2, 0.6],
            y= [-0.25, 0.25],
        )
        init_base_rot_range = dict(
            roll= [-0.1, 0.1],
            pitch= [-0.1, 0.1],
        )

        push_robots = False

    class rewards( AliengoFieldCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.1
            world_vel_l2norm = -1.
            legs_energy_substeps = -1e-6
            alive = 2.
            penetrate_depth = 0#-1e-2
            penetrate_volume = 0#-1e-2
            exceed_dof_pos_limits = -1e-1
            exceed_torque_limits_i = -2e-1
        soft_dof_pos_limit = 0.8
        max_contact_force = 100.0

    class curriculum( AliengoFieldCfg.curriculum ):
        penetrate_volume_threshold_harder = 8000
        penetrate_volume_threshold_easier = 12000
        penetrate_depth_threshold_harder = 1000
        penetrate_depth_threshold_easier = 1600


logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class AliengoJumpCfgPPO( AliengoFieldCfgPPO ):
    seed=1
    class algorithm( AliengoFieldCfgPPO.algorithm ):
        add_skill_discovery_loss = False
        entropy_coef = 0.0
        clip_min_std = 0.2
    
    class runner( AliengoFieldCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        experiment_name = "field_aliengo"
        resume = False
        # load_run = "{Your traind walking model directory}"
        # load_run = "{Your virtually trained jump model directory}"
        
        # run_name = "".join(["Skills_",
        # ("jump" if AliengoJumpCfg.terrain.BarrierTrack_kwargs["jump"]["jump_down_prob"] < 1. else "down"),
        # ("_virtual" if AliengoJumpCfg.terrain.BarrierTrack_kwargs["virtual_terrain"] else ""),
        # ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        # ])
        max_iterations = 20000
        save_interval = 1000
    