import torch
import torch.nn as nn
from .actor_critic import ActorCritic, get_activation
import torch.nn.functional as F
from torch.distributions import Normal


class ActorCriticMetra(ActorCritic):
    is_recurrent = False

    def __init__(self, num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 discriminator_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 mu_activation=None,
                 **kwargs):
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                if mu_activation:
                    actor_layers.append(get_activation(mu_activation))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.div_critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Critic MLP: {self.div_critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        self.skill_dim = kwargs['skill_dim']
        self.input_dim = kwargs['phi_input_dim']

        discriminator_layers = []
        discriminator_layers.append(nn.Linear(self.input_dim, discriminator_hidden_dims[0]))
        discriminator_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                discriminator_layers.append(nn.Linear(discriminator_hidden_dims[l], self.skill_dim))
            else:
                discriminator_layers.append(nn.Linear(discriminator_hidden_dims[l], discriminator_hidden_dims[l + 1]))
                discriminator_layers.append(activation)
        self.discriminator = nn.Sequential(*discriminator_layers)

        self.lamda = torch.nn.Parameter(torch.tensor([30.]))
        self.epsilon = 0.001



        print(f"Discriminator MLP: {self.discriminator}")


    # def discriminator_inference(self, observations):
    #     # for discretization experiment
    #
    #     pos = observations[:,:2]
    #
    #     pos_int = (pos[:, :2] * 4).long()
    #     pos_int[:, 0] += 14  # x:[-1~3] y : [1~-3]
    #     pos_int[:, 1] += 14
    #     pos_int = torch.clip(pos_int, min=0, max=27)
    #     pos_int = pos_int.float()/14
    #
    #
    #     return self.discriminator(pos_int)

    def discriminator_inference(self, observations):
        return self.discriminator(observations[:, 0:self.input_dim])

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        div_value = self.div_critic(critic_observations)

        return value, div_value

