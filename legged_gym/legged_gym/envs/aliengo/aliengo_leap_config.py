import numpy as np
from legged_gym.envs.aliengo.aliengo_field_config import AliengoFieldCfg, AliengoFieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class AliengoLeapCfg( AliengoFieldCfg ):

    #### uncomment this to train non-virtual terrain
    # class sensor( AliengoFieldCfg.sensor ):
    #     class proprioception( AliengoFieldCfg.sensor.proprioception ):
    #         delay_action_obs = True
    #         latency_range = [0.04-0.0025, 0.04+0.0075]
    #### uncomment the above to train non-virtual terrain
    
    class terrain( AliengoFieldCfg.terrain ):
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.
        curriculum = False

        BarrierTrack_kwargs = merge_dict(AliengoFieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "leap",
            ],
            leap= dict(
                length= (0.48, 0.48),
                depth= (0.4, 0.8),
                height= 0.2,
            ),
            virtual_terrain= False, # Change this to False for real terrain
            no_perlin_threshold= 0.06,
        ))

        TerrainPerlin_kwargs = merge_dict(AliengoFieldCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= [0.05, 0.1],
        ))
    
    class commands( AliengoFieldCfg.commands ):
        class ranges( AliengoFieldCfg.commands.ranges ):
            lin_vel_x = [1.0, 1.5]
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
        roll_kwargs = merge_dict(AliengoFieldCfg.termination.roll_kwargs, dict(
            threshold= 0.4,
            leap_threshold= 0.4,
        ))
        z_high_kwargs = merge_dict(AliengoFieldCfg.termination.z_high_kwargs, dict(
            threshold= 2.0,
        ))
    class control( AliengoFieldCfg.control ):
        stiffness = {'joint': 50.}
        damping = {'joint': 1.}
        action_scale = 0.5
        torque_limits = 25 # override the urdf
        computer_clip_torque = True
        motor_clip_torque = True
        
    class domain_rand(AliengoFieldCfg.domain_rand):
        # push_robots = True # use for virtual training
        push_robots = False  # use for non-virtual training
    class rewards( AliengoFieldCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.05
            world_vel_l2norm = -1.
            legs_energy_substeps = -1e-6
            alive = 2.
            penetrate_depth = 0 #-4e-3
            penetrate_volume = 0 #-4e-3
            exceed_dof_pos_limits = -1e-1
            exceed_torque_limits_i = -2e-1

    class curriculum( AliengoFieldCfg.curriculum ):
        penetrate_volume_threshold_harder = 9000
        penetrate_volume_threshold_easier = 10000
        penetrate_depth_threshold_harder = 300
        penetrate_depth_threshold_easier = 5000


class AliengoLeapCfgPPO( AliengoFieldCfgPPO ):
    seed=4
    class algorithm( AliengoFieldCfgPPO.algorithm ):
        entropy_coef = 0.0
        clip_min_std = 0.2
        add_skill_discovery_loss = False
        kappa = 0.0
    class runner( AliengoFieldCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        experiment_name = "field_aliengo"
        # run_name = "".join(["Skill",
        # ("Multi" if len(AliengoLeapCfg.terrain.BarrierTrack_kwargs["options"]) > 1 else (AliengoLeapCfg.terrain.BarrierTrack_kwargs["options"][0] if AliengoLeapCfg.terrain.BarrierTrack_kwargs["options"] else "PlaneWalking")),
        # ("_propDelay{:.2f}-{:.2f}".format(
        #         AliengoLeapCfg.sensor.proprioception.latency_range[0],
        #         AliengoLeapCfg.sensor.proprioception.latency_range[1],
        #     ) if AliengoLeapCfg.sensor.proprioception.delay_action_obs else ""
        # ),
        # ("_pEnergySubsteps{:.0e}".format(AliengoLeapCfg.rewards.scales.legs_energy_substeps) if AliengoLeapCfg.rewards.scales.legs_energy_substeps != -2e-6 else ""),
        # ("_virtual" if AliengoLeapCfg.terrain.BarrierTrack_kwargs["virtual_terrain"] else ""),
        # ])
        resume = False
        # load_run = "{Your traind walking model directory}"
        # load_run = "{Your virtually trained leap model directory}"
        max_iterations = 10000
        save_interval = 1000
    