import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic, ActorCriticMetra
from rsl_rl.storage import RolloutStorage

from .ppo import PPO


class PPOMetra(PPO):
    # actor_critic: ActorCriticMetra

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device, True)

    def process_env_step(self, next_obs, rewards, dones, infos):
        self.transition.next_observations = next_obs
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)


    def compute_losses(self, minibatch):
        self.actor_critic.act(minibatch.obs, masks=minibatch.masks, hidden_states=minibatch.hid_states[0])
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(minibatch.actions)
        value_batch = self.actor_critic.evaluate(minibatch.critic_obs, masks=minibatch.masks,
                                                 hidden_states=minibatch.hid_states[1])
        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        try:
            entropy_batch = self.actor_critic.entropy
        except:
            entropy_batch = None

        # KL
        if self.desired_kl != None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / minibatch.old_sigma + 1.e-5) + (
                                torch.square(minibatch.old_sigma) + torch.square(minibatch.old_mu - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(minibatch.old_actions_log_prob))
        surrogate = -torch.squeeze(minibatch.advantages) * ratio
        surrogate_clipped = -torch.squeeze(minibatch.advantages) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = minibatch.values + (value_batch - minibatch.values).clamp(-self.clip_param,
                                                                                      self.clip_param)
            value_losses = (value_batch - minibatch.returns).pow(2)
            value_losses_clipped = (value_clipped - minibatch.returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (minibatch.returns - value_batch).pow(2).mean()

        phi_next_obs = self.actor_critic.discriminator_inference(minibatch.next_obs)
        phi_obs = self.actor_critic.discriminator_inference(minibatch.obs)
        z = minibatch.obs[:, -self.actor_critic.skill_dim:]

        phi_diff = phi_next_obs - phi_obs
        phi_norm_squared = torch.square(torch.norm((phi_diff), dim=1, p=2))

        phi_loss = (phi_diff * z).sum(dim=1) + \
                   self.actor_critic.lamda.detach() * torch.clamp(1. - phi_norm_squared,
                                                                  max=self.actor_critic.epsilon)
        phi_loss = -(phi_loss * (1 - minibatch.dones)).mean()
        # phi_loss = -(phi_loss * dones).mean() # for last state only
        lamda_loss = self.actor_critic.lamda * torch.clamp(1. - phi_norm_squared.detach(),
                                                           max=self.actor_critic.epsilon)
        lamda_loss = (lamda_loss * (1 - minibatch.dones)).mean()
        # lamda_loss = lamda_loss * dones # for last state only




        return_ = dict(
            surrogate_loss=surrogate_loss,
            value_loss=value_loss,
            phi_loss=phi_loss,
            lamda_loss=lamda_loss,
        )
        if entropy_batch is not None:
            return_["entropy"] = - entropy_batch.mean()

        inter_vars = dict(
            ratio=ratio,
            surrogate=surrogate,
            surrogate_clipped=surrogate_clipped,
        )
        if self.desired_kl != None and self.schedule == 'adaptive':
            inter_vars["kl"] = kl
        if self.use_clipped_value_loss:
            inter_vars["value_clipped"] = value_clipped
        return return_, inter_vars, dict()
