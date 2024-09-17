import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic, ActorCriticMetra
from rsl_rl.storage import RolloutStorage

from .ppo import PPO
from collections import defaultdict

from torch.autograd import Variable




class PPOMetra(PPO):
    # actor_critic: ActorCriticMetra
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 clip_min_std=1e-15,  # clip the policy.std if it supports, check update()
                 optimizer_class_name="Adam",
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 add_skill_discovery_loss=False,
                 add_next_state=False,
                 skill_reward_coef=0.0,
                 adjustable_kappa=False,
                 kappa_cap=1.0,
                 kappa=0.0
                 ):
        super().__init__(actor_critic,
                 num_learning_epochs=num_learning_epochs,
                 num_mini_batches=num_mini_batches,
                 clip_param=clip_param,
                 gamma=gamma,
                 lam=lam,
                 value_loss_coef=value_loss_coef,
                 entropy_coef=entropy_coef,
                 learning_rate=learning_rate,
                 max_grad_norm=max_grad_norm,
                 use_clipped_value_loss=use_clipped_value_loss,
                 clip_min_std=clip_min_std,  # clip the policy.std if it supports, check update()
                 optimizer_class_name=optimizer_class_name,
                 schedule=schedule,
                 desired_kl=desired_kl,
                 device=device,
                 add_skill_discovery_loss=add_skill_discovery_loss,
                 add_next_state=add_next_state,
                 skill_reward_coef=skill_reward_coef,
                 adjustable_kappa=adjustable_kappa,
                 kappa_cap=kappa_cap,
                 kappa=kappa
                 )
        self.adjustable_kappa = adjustable_kappa
        self.kappa = torch.tensor([kappa], dtype=torch.float, device=device)
        self.kappa_cap = kappa_cap
        if adjustable_kappa:
            actor_num_params = sum(p.numel() for p in self.actor_critic.actor.parameters())
            self.prev_grad = torch.zeros(actor_num_params, dtype=torch.float, device=device)




    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device, True)

    def process_env_step(self, next_obs, rewards, div_rewards, dones, infos):
        self.transition.next_observations = next_obs
        self.transition.rewards = rewards.clone()
        self.transition.div_rewards = div_rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards     += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            self.transition.div_rewards += self.gamma * torch.squeeze(self.transition.div_values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)


    def compute_losses(self, minibatch):
        self.actor_critic.act(minibatch.obs, masks=minibatch.masks, hidden_states=minibatch.hid_states[0])
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(minibatch.actions)
        value_batch, div_value_batch = self.actor_critic.evaluate(minibatch.critic_obs, masks=minibatch.masks,
                                                 hidden_states=minibatch.hid_states[1])


        if self.adjustable_kappa:
            adv = minibatch.advantages + self.kappa * minibatch.div_advantages
        else:
            adv = minibatch.advantages + minibatch.div_advantages

        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        try:
            entropy_batch = self.actor_critic.entropy
        except:
            entropy_batch = None


        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(minibatch.old_actions_log_prob))
        surrogate = -torch.squeeze(adv) * ratio
        surrogate_clipped = -torch.squeeze(adv) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        if self.adjustable_kappa:
            surrogate_div = -torch.squeeze(minibatch.div_advantages) * ratio
            surrogate_clipped_div = -torch.squeeze(minibatch.div_advantages) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss_div = torch.max(surrogate_div, surrogate_clipped_div).mean()
            surrogate_loss_div.backward(retain_graph=True)
            actor_grad_div = self._get_actor_grads()
            for param in self.actor_critic.actor.parameters():
                param.grad = None


        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = minibatch.values + (value_batch - minibatch.values).clamp(-self.clip_param,
                                                                                      self.clip_param)
            value_losses = (value_batch - minibatch.returns).pow(2)
            value_losses_clipped = (value_clipped - minibatch.returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()

            div_value_clipped = minibatch.div_values + (div_value_batch - minibatch.div_values).clamp(-self.clip_param,
                                                                                      self.clip_param)
            div_value_losses = (div_value_batch - minibatch.div_returns).pow(2)
            div_value_losses_clipped = (div_value_clipped - minibatch.div_returns).pow(2)
            div_value_loss = torch.max(div_value_losses, div_value_losses_clipped).mean()
        else:
            value_loss = (minibatch.returns - value_batch).pow(2).mean()
            div_value_loss = (minibatch.div_returns - div_value_batch).pow(2).mean()

        value_loss = value_loss + self.kappa.item() * div_value_loss

        phi_next_obs = self.actor_critic.discriminator_inference(minibatch.next_obs)
        phi_obs = self.actor_critic.discriminator_inference(minibatch.obs)
        z = minibatch.obs[:, -self.actor_critic.skill_dim:]

        phi_diff = phi_next_obs - phi_obs
        phi_norm_squared = torch.square(torch.norm((phi_diff), dim=1, p=2))

        phi_loss = (phi_diff * z).sum(dim=1) + \
                   self.actor_critic.lamda.detach() * torch.clamp(1. - phi_norm_squared,
                                                                  max=self.actor_critic.epsilon)
        phi_loss = -(phi_loss * (1 - minibatch.dones)).mean()
        lamda_loss = self.actor_critic.lamda * torch.clamp(1. - phi_norm_squared.detach(),
                                                           max=self.actor_critic.epsilon)
        lamda_loss = (lamda_loss * (1 - minibatch.dones)).mean()


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
        if self.adjustable_kappa:
            return return_, inter_vars, dict(actor_grad_div=actor_grad_div)
        else: 
            return return_, inter_vars, dict()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        value, div_value = self.actor_critic.evaluate(critic_obs)
        self.transition.values = value.detach()
        self.transition.div_values = div_value.detach()

        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def compute_returns(self, last_critic_obs):
        last_values, last_div_values = self.actor_critic.evaluate(last_critic_obs)
        last_values = last_values.detach()
        last_div_values = last_div_values.detach()

        self.storage.compute_div_mixed_returns(last_values, last_div_values, self.gamma, self.lam)


    def update(self, current_learning_iteration):
        self.current_learning_iteration = current_learning_iteration
        mean_losses = defaultdict(lambda :0.)
        average_stats = defaultdict(lambda :0.)
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for minibatch in generator:

                losses, _, stats = self.compute_losses(minibatch)

                loss = 0.
                for k, v in losses.items():
                    loss += getattr(self, k + "_coef", 1.) * v
                    mean_losses[k] = mean_losses[k] + v.detach()
                mean_losses["total_loss"] = mean_losses["total_loss"] + loss.detach()
                for k, v in stats.items():
                    average_stats[k] = average_stats[k] + v.detach()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                if self.adjustable_kappa:
                    actor_grad = self._get_actor_grads()
                    self.update_kappa(actor_grad)
                    self.prev_grad.copy_(stats["actor_grad_div"])

                self.optimizer.step()
            
        num_updates = self.num_learning_epochs * self.num_mini_batches
        for k in mean_losses.keys():
            mean_losses[k] = mean_losses[k] / num_updates
        for k in average_stats.keys():
            average_stats[k] = average_stats[k] / num_updates
        self.storage.clear()
        if hasattr(self.actor_critic, "clip_std"):
            self.actor_critic.clip_std(min= self.clip_min_std)

        return mean_losses, average_stats

    def update_kappa(self, actor_grad):
        x = self.prev_grad * actor_grad
        self.kappa += x.sum() * 0.001
        self.kappa = torch.clamp(self.kappa, min=0., max=self.kappa_cap)
        # import ipdb;ipdb.set_trace()
        # print (self.kappa_cap)

    def _get_actor_grads(self):
        grads = []
        for param in self.actor_critic.actor.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        x = torch.clone(grads)
        return x
