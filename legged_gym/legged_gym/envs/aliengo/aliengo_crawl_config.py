import numpy as np
from legged_gym.envs.aliengo.aliengo_field_config import AliengoFieldCfg, AliengoFieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class AliengoCrawlCfg( AliengoFieldCfg ):

    #### uncomment this to train non-virtual terrain
    class sensor( AliengoFieldCfg.sensor ):
        class proprioception( AliengoFieldCfg.sensor.proprioception ):
            delay_action_obs = False
            latency_range = [0.0, 0.0]
    #### uncomment the above to train non-virtual terrain
    
    class terrain( AliengoFieldCfg.terrain ):
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.
        curriculum = False

        BarrierTrack_kwargs = merge_dict(AliengoFieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "crawl",
            ],
            track_block_length= 1.6,
            crawl= dict(
                height= (0.3, 0.3),
                depth= (0.1, 0.6), # size along the forward axis
                wall_height= 0.6,
                no_perlin_at_obstacle= False,
            ),
            virtual_terrain= False, # Change this to False for real terrain
        ))

        TerrainPerlin_kwargs = merge_dict(AliengoFieldCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= 0.1,
        ))
    
    class commands( AliengoFieldCfg.commands ):
        class ranges( AliengoFieldCfg.commands.ranges ):
            lin_vel_x = [0.3, 0.8]
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

    class rewards( AliengoFieldCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.05
            world_vel_l2norm = -1.
            legs_energy_substeps = -2e-5
            alive = 2.
            penetrate_depth = -6e-2 # comment this out if trianing non-virtual terrain
            penetrate_volume = -6e-2 # comment this out if trianing non-virtual terrain
            exceed_dof_pos_limits = -1e-1
            exceed_torque_limits_i = -2e-1

    class curriculum( AliengoFieldCfg.curriculum ):
        penetrate_volume_threshold_harder = 1500
        penetrate_volume_threshold_easier = 10000
        penetrate_depth_threshold_harder = 10
        penetrate_depth_threshold_easier = 400


class AliengoCrawlCfgPPO( AliengoFieldCfgPPO ):
    class algorithm( AliengoFieldCfgPPO.algorithm ):
        add_skill_discovery_loss = False
        add_next_state = False
        entropy_coef = 0.0
        clip_min_std = 0.2
    
    class runner( AliengoFieldCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        experiment_name = "field_aliengo"
        run_name = "".join(["Skill",
        ("Multi" if len(AliengoCrawlCfg.terrain.BarrierTrack_kwargs["options"]) > 1 else (AliengoCrawlCfg.terrain.BarrierTrack_kwargs["options"][0] if AliengoCrawlCfg.terrain.BarrierTrack_kwargs["options"] else "PlaneWalking")),
        ("_propDelay{:.2f}-{:.2f}".format(
                AliengoCrawlCfg.sensor.proprioception.latency_range[0],
                AliengoCrawlCfg.sensor.proprioception.latency_range[1],
            ) if AliengoCrawlCfg.sensor.proprioception.delay_action_obs else ""
        ),
        ("_pEnergy" + np.format_float_scientific(AliengoCrawlCfg.rewards.scales.legs_energy_substeps, precision= 1, exp_digits= 1, trim= "-") if AliengoCrawlCfg.rewards.scales.legs_energy_substeps != 0. else ""),
        ("_virtual" if AliengoCrawlCfg.terrain.BarrierTrack_kwargs["virtual_terrain"] else ""),
        ])
        resume = False
        load_run = "{Your traind walking model directory}"
        load_run = "{Your virtually trained crawling model directory}"
        max_iterations = 20000
        save_interval = 500
    