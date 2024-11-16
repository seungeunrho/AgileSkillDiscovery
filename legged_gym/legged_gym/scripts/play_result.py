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
import os
os.environ["PATH"] = "/home/serho/miniconda3/envs/parkour/bin:/home/serho/miniconda3/condabin:" + os.environ["PATH"]
os.environ["PATH"] = "/home/srho31/miniconda3/envs/isaac/bin:/home/srho31/miniconda3/condabin:" + os.environ["PATH"]


from legged_gym import LEGGED_GYM_ROOT_DIR
from collections import OrderedDict
import os
import json
import time
import numpy as np
np.float = np.float32
import isaacgym
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import update_class_from_dict
from legged_gym.utils.observation import get_obs_slice
from legged_gym.debugger import break_into_debugger
from tqdm import tqdm
import numpy as np
import torch


def create_recording_camera(gym, env_handle,
        resolution= (1920, 1080),
        h_fov= 86,
        actor_to_attach= None,
        transform= None, # related to actor_to_attach
    ):
    camera_props = gymapi.CameraProperties()
    camera_props.enable_tensors = True
    camera_props.width = resolution[0]
    camera_props.height = resolution[1]
    camera_props.horizontal_fov = h_fov
    camera_handle = gym.create_camera_sensor(env_handle, camera_props)
    if actor_to_attach is not None:
        gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            actor_to_attach,
            transform,
            gymapi.FOLLOW_POSITION,
        )
    elif transform is not None:
        gym.set_camera_transform(
            camera_handle,
            env_handle,
            transform,
        )
    return camera_handle

@torch.no_grad()
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    print("max episode length: ",env_cfg.env.episode_length_s)
    # import ipdb;ipdb.set_trace()
    # print(os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, args.load_run, "config.json"))
    with open(os.path.join(LEGGED_GYM_ROOT_DIR,"logs", train_cfg.runner.experiment_name, args.load_run, "config.json"), "r") as f:
        d = json.load(f, object_pairs_hook= OrderedDict)
        update_class_from_dict(env_cfg, d, strict= True)
        update_class_from_dict(train_cfg, d, strict= True)

    # override some parameters for testing
    if env_cfg.terrain.selected == "BarrierTrack":
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
        env_cfg.env.episode_length_s = 20
        
        env_cfg.terrain.max_init_terrain_level = 0
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
    else:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
        env_cfg.env.episode_length_s = 60
        env_cfg.terrain.terrain_length = 8
        env_cfg.terrain.terrain_width = 8
        env_cfg.terrain.max_init_terrain_level = 0
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    # env_cfg.terrain.BarrierTrack_kwargs["options"] = [
    #     "crawl",
    #     # "jump",
    #     # "leap",
    #     # "tilt",
    # ]
    if "one_obstacle_per_track" in env_cfg.terrain.BarrierTrack_kwargs.keys():
        env_cfg.terrain.BarrierTrack_kwargs.pop("one_obstacle_per_track")
    num_obstacles = 3
    env_cfg.terrain.BarrierTrack_kwargs["n_obstacles_per_track"] = num_obstacles   
    # import ipdb;ipdb.set_trace() 
    if train_cfg.runner.experiment_name =='a1_crawl_metra':
        env_cfg.commands.ranges.lin_vel_x = [0.6, 0.6]
    else:
        env_cfg.commands.ranges.lin_vel_x = [1.2, 1.2]
    if "distill" in args.task:
        env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
        env_cfg.commands.ranges.lin_vel_y = [-0., 0.]
        env_cfg.commands.ranges.ang_vel_yaw = [-0., 0.]
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.init_base_pos_range = dict(
        x= [0.6, 0.6],
        y= [-0.05, 0.05],
    )
    env_cfg.termination.termination_terms = []
    env_cfg.termination.timeout_at_border = False
    env_cfg.termination.timeout_at_finished = False
    env_cfg.viewer.debug_viz = False # in a1_distill, setting this to true will constantly showing the egocentric depth view.
    env_cfg.viewer.draw_volume_sample_points = False
    train_cfg.runner.resume = True
    train_cfg.runner_class_name = "OnPolicyRunner"
    if "distill" in args.task: # to save the memory
        train_cfg.algorithm.teacher_policy.sub_policy_paths = []
        train_cfg.algorithm.teacher_policy_class_name = "ActorCritic"
        train_cfg.algorithm.teacher_policy = dict(
            num_actor_obs= 48,
            num_critic_obs= 48,
            num_actions= 12,
        )

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # register debugging options to manually trigger disruption
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_P, "push_robot")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_L, "press_robot")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_J, "action_jitter")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_Q, "exit")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_R, "agent_full_reset")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_U, "full_reset")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_C, "resample_commands")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_W, "forward")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_S, "backward")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_A, "leftward")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_D, "rightward")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_F, "leftturn")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_G, "rightturn")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_X, "stop")
    # load policy
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        save_cfg= False,
    )
    agent_model = ppo_runner.alg.actor_critic
    policy = ppo_runner.get_inference_policy(device=env.device)
    ### get obs_slice to read the obs
    # obs_slice = get_obs_slice(env.obs_segments, "engaging_block")

    if train_cfg.algorithm.add_skill_discovery_loss:
        env.set_actor_in_env(ppo_runner.alg.actor_critic)

    env.reset()
    print("terrain_levels:", env.terrain_levels.float().mean(), env.terrain_levels.float().max(),
          env.terrain_levels.float().min())
    obs = env.get_observations()
    critic_obs = env.get_privileged_observations()
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
    if RECORD_FRAMES:
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(*env_cfg.viewer.pos)
        transform.r = gymapi.Quat.from_euler_zyx(0., 0., -np.pi/2)
        recording_camera = create_recording_camera(
            env.gym,
            env.envs[0],
            transform= transform,
        )

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 4 # which joint is used for logging
    stop_state_log = 512 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([0.6, 0., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    # import ipdb;ipdb.set_trace()
    # camera_direction = np.array([0,2.0,-0.5])
    camera_follow_id = 0 # only effective when CAMERA_FOLLOW
    img_idx = 0
    num_iter = 30/ env_cfg.env.num_envs
    n_obstacles_passed = []
    n_obstacles_passed_2 = torch.empty((0,1))
    n_obstacles_passed_3 = torch.empty((0,1))
    if hasattr(env, "motor_strength"):
        print("motor_strength:", env.motor_strength[robot_index].cpu().numpy().tolist())
    print("torque_limits:", env.torque_limits)
    start_time = time.time_ns()
    while len(n_obstacles_passed_2) <1000:
        for i in tqdm(range(1*int(env.max_episode_length))):
            if "obs_slice" in locals().keys():
                obs_component = obs[:, obs_slice[0]].reshape(-1, *obs_slice[1])
            
            actions = policy(obs.detach())
            teacher_actions = actions
            obs, critic_obs, rews, dones, infos = env.step(actions.detach())
            # print(env.extras['episode']['n_obstacle_passed'])
            if i < stop_state_log:
                if torch.is_tensor(env.cfg.control.action_scale):
                    action_scale = env.cfg.control.action_scale.detach().cpu().numpy()[joint_index]
                else:
                    action_scale = env.cfg.control.action_scale
                base_roll = get_euler_xyz(env.base_quat)[0][robot_index].item()
                base_pitch = get_euler_xyz(env.base_quat)[1][robot_index].item()
                if base_pitch > torch.pi: base_pitch -= torch.pi * 2
   
            elif i==stop_state_log:
                logger.plot_states()
                env._get_terrain_curriculum_move(torch.tensor([0], device= env.device))
            if  0 < i < stop_rew_log:
                if infos["episode"]:
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes>0:
                        logger.log_rewards(infos["episode"], num_episodes)
            elif i==stop_rew_log:
                logger.print_rewards()
            
            # if dones.any():
            if dones.any():
                with torch.no_grad():
                    pos_x = env.extras['episode']['pos_x'] 
                    
                    n_obstacle_passed_ = torch.clip(
                        torch.div(pos_x, env.terrain.env_block_length, rounding_mode= "floor") - 1,
                        min= 0.0,
                    ).cpu()
                    
            
                agent_model.reset(dones)
                n_obstacles_passed.append(env.extras['episode']['n_obstacle_passed'])
                # import ipdb;ipdb.set_trace()
                n_obstacles_passed_2 = torch.cat((n_obstacles_passed_2,torch.clamp(env.extras['episode']['n_obstacle_passed_per_robot'].unsqueeze(1),min=0,max=num_obstacles)),0)
                n_obstacles_passed_3 = torch.cat((n_obstacles_passed_3,n_obstacle_passed_.unsqueeze(1)),0)

                # n_obstacles_passed_3.append(n_obstacle_passed_)
                print("env dones,{} because has timeout".format("" if env.time_out_buf[dones].any() else " not"))
                print(infos)
                
                # break
                # continue
            # if i % 100 == 0:
            #     print("frame_rate:" , 100/(time.time_ns() - start_time) * 1e9, 
            #         "command_x:", env.commands[robot_index, 0],
            #     )
                start_time = time.time_ns()
           
        n_obstacles_passed.append(env.extras['episode']['n_obstacle_passed'])    
        # import ipdb;ipdb.set_trace()
    print("Mean: ", n_obstacles_passed_2[:1000].mean())
    print("Std: ", n_obstacles_passed_2[:1000].std())
    file_name = train_cfg.runner.experiment_name+"_"+str(train_cfg.seed)+"_"+str(train_cfg.runner.checkpoint)+ "_n_obstacle_"+str(num_obstacles)+".pt"
    # import ipdb;ipdb.set_trace()
    torch.save(n_obstacles_passed_2, file_name)
    
    import ipdb;ipdb.set_trace()
if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    CAMERA_FOLLOW = False
    args = get_args()
    play(args)
