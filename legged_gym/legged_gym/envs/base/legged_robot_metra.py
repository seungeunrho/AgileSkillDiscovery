import torch
from .legged_robot_noisy import LeggedRobotNoisy
from .legged_robot import LeggedRobot
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict, defaultdict

import torch.nn as nn
from isaacgym.torch_utils import *

from collections import OrderedDict, defaultdict
import torch

from legged_gym.utils.terrain import get_terrain_cls


class LeggedRobotMetra(LeggedRobotNoisy):
    def _init_buffers(self):
        super()._init_buffers()
        self.skill_dim = self.cfg.env.skill_dim
        self.phi_input_dim = self.cfg.env.phi_input_dim
        self.skills = torch.zeros(self.num_envs, self.skill_dim, device=self.device, dtype=torch.float, requires_grad=False)
        self.prev_skill_obs_buf = torch.zeros(self.num_envs, self.phi_input_dim, device=self.device, dtype=torch.float, requires_grad=False)

        # self.sample_skill = self.cfg.env.sample_skill


    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.actions[env_ids] = 0.

        self.obs_buf[env_ids] = 0.
        self.prev_skill_obs_buf[env_ids] = 0.

        if self.cfg.env.sample_skill:
            self.skills[env_ids] = torch.normal(0, 1, size=(len(env_ids), self.skill_dim)).to(self.device)


        # if len(env_ids)>0:
        #     print("cur skill", self.skills)
        
    def get_obs_segment_from_components(self, components):
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        segments = OrderedDict()
        if "base_pose" in components:
            segments["base_pose"] = (6,) # xyz + rpy
        if "proprioception" in components:
            segments["proprioception"] = (48,)
        if "height_measurements" in components:
            segments["height_measurements"] = (187,)
        if "forward_depth" in components:
            segments["forward_depth"] = (1, *self.cfg.sensor.forward_camera.resolution)
        
        if "robot_config" in components:
            """ Related to robot_config_buffer attribute, Be careful to change. """
            # robot shape friction
            # CoM (Center of Mass) x, y, z
            # base mass (payload)
            # motor strength for each joint
            segments["robot_config"] = (1 + 3 + 1 + 12,)
        if "engaging_block" in components:
            if not self.check_BarrierTrack_terrain():
                # This could be wrong, please check the implementation of BarrierTrack
                segments["engaging_block"] = (1 + (4 + 1) + 2,)
            else:
                segments["engaging_block"] = get_terrain_cls("BarrierTrack").get_engaging_block_info_shape(self.cfg.terrain)
        if "sidewall_distance" in components:
            self.check_BarrierTrack_terrain()
            segments["sidewall_distance"] = (2,)
        if "skills" in components:
            segments["skills"] = (self.cfg.env.skill_dim,)
        return segments

    def compute_observations(self):
        self.prev_skill_obs_buf[:] = self.obs_buf[:, :self.phi_input_dim]
        super().compute_observations()

    def set_actor_in_env(self, actor_critic):
        self.actor_critic = actor_critic

    def fix_skill(self, skill_vec):
        # for only playing with one agent
        self.skills[0] = torch.tensor(skill_vec).to(self.device)

    def _reward_diversity(self):
        with torch.no_grad():
            phi_s_prime = self.actor_critic.discriminator_inference(self.obs_buf)
            phi_s = self.actor_critic.discriminator_inference(self.prev_skill_obs_buf)
            int_rew = (phi_s_prime - phi_s) * self.skills
            int_rew = torch.sum(int_rew, dim=-1)
            # int_rew = (phi_s_prime - phi_s) * self.skills / torch.norm(self.skills, p=2)

            # x = phi_s_prime - phi_s
            #
            # if torch.norm(x, dim=-1, p=2).max() > 5.0:
            #     print(torch.norm(x, dim=-1, p=2).max())
            #     aa=1


            is_zero = (torch.sum(self.prev_skill_obs_buf, dim=-1) == 0)
            int_rew = int_rew * (1-is_zero.float())


        return int_rew

    def _write_skills_noise(self, noise_vec):
        # noise_vec[:] = self.cfg.noise.noise_scales.forward_depth * self.cfg.noise.noise_level * self.obs_scales.forward_depth
        pass

    def _get_skills_obs(self, privileged=False):
        return self.skills
