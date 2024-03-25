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
                 **kwargs):
        super().__init__(num_actor_obs,
                         num_critic_obs,
                         num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         discriminator_hidden_dims=discriminator_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std,
                         **kwargs)

        self.skill_dim = kwargs['skill_dim']
        self.input_dim = kwargs['phi_input_dim']

        activation = get_activation(activation)
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

