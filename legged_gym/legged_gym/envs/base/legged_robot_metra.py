import torch
from .legged_robot_noisy import LeggedRobotNoisy

from collections import OrderedDict, defaultdict


from legged_gym.utils.terrain import get_terrain_cls

from isaacgym.torch_utils import *
from isaacgym import gymtorch

import torch
from legged_gym.envs.base.base_task import BaseTask


class LeggedRobotMetra(LeggedRobotNoisy):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.div_rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def _init_buffers(self):
        super()._init_buffers()
        self.skill_dim = self.cfg.env.skill_dim
        self.phi_start_dim = self.cfg.env.phi_start_dim
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
        self.prev_skill_obs_buf[:] = self.obs_buf[:, self.phi_start_dim : self.phi_start_dim + self.phi_input_dim]
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


    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.div_rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            if name == "diversity":
                div_rew = self.reward_functions[i]() * self.reward_scales[name]
                self.div_rew_buf += div_rew
                self.episode_sums[name] += div_rew
            else:
                rew = self.reward_functions[i]() * self.reward_scales[name]
                self.rew_buf += rew
                self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew


    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.pre_physics_step(actions)
        # step physics and render each frame
        self.render()
        for dec_i in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.post_decimation_step(dec_i)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, (self.rew_buf, self.div_rew_buf), self.reset_buf, self.extras

