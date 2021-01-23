#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import torch.nn as nn
from torch.nn import functional as F
from Algorithms.utils import to_tensor, layer_init
from Algorithms.body import *

class SingleOptionNet(nn.Module):
    def __init__(self,
                 action_dim,
                 body_fn):
        super(SingleOptionNet, self).__init__()
        self.pi_body = body_fn()
        self.beta_body = body_fn()
        self.fc_pi = layer_init(nn.Linear(self.pi_body.latent_dim, action_dim), 1e-3)
        self.fc_beta = layer_init(nn.Linear(self.beta_body.latent_dim, 1), 1e-3)
        self.std = nn.Parameter(torch.zeros((1, action_dim)))

    def forward(self, phi):
        phi_pi = self.pi_body(phi)
        mean = torch.tanh(self.fc_pi(phi_pi))
        std = F.softplus(self.std).expand(mean.size(0), -1)

        phi_beta = self.beta_body(phi)
        beta = torch.sigmoid(self.fc_beta(phi_beta))

        return {
            'mean': mean,
            'std': std,
            'beta': beta,
        }

class OptionGaussianActorCriticNet(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 num_options,
                 phi_body=None,
                 critic_body=None,
                 option_body_fn=None,
                 device='cpu'):
        '''
        Option_Critic network for continuous action space.
        Architecture:
            phi_body: encoder of sorts, environment observation will pass through phi_body 
                        before going into policy and critic networks
            options: Consist of a list of SingleOptionNet, 
                    where they form the lower level policy network to output actions, 
                    as well as termination for each option
            fc_q: Network for state option q values
        '''
        super(OptionGaussianActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.latent_dim)
        if option_body_fn is None: option_body_fn = lambda: DummyBody(phi_body.latent_dim)

        self.phi_body = phi_body
        self.critic_body = critic_body

        # build option network
        self.options = nn.ModuleList([SingleOptionNet(action_dim, option_body_fn) for _ in range(num_options)])

        # linear output
        self.fc_q_o = layer_init(nn.Linear(critic_body.latent_dim, num_options), 1e-3)

        self.num_options = num_options
        self.action_dim = action_dim
        self.device = device
        self.to(device)

    def forward(self, obs):

        # state feature
        obs = to_tensor(obs).unsqueeze(0).to(self.device)
        phi = self.phi_body(obs)

        # option
        mean = []
        std = []
        beta = []
        for option in self.options:
            prediction = option(phi)
            mean.append(prediction['mean'].unsqueeze(1))
            std.append(prediction['std'].unsqueeze(1))
            beta.append(prediction['beta'])
        mean = torch.cat(mean, dim=1)
        std = torch.cat(std, dim=1)
        beta = torch.cat(beta, dim=1)

        # critic network
        phi_c = self.critic_body(phi)
        q_o = self.fc_q_o(phi_c)

        return {'mean': mean,
                'std': std,
                'q_o': q_o,
                'beta': beta}

class OptionCriticNet(nn.Module):
    def __init__(self, body, action_dim, num_options, device='cpu'):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.latent_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.latent_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.latent_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.device = device
        self.to(device)

    def forward(self, x):
        phi = self.body(to_tensor(x).to(self.device))
        q = self.fc_q(phi)
        beta = torch.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        pi = F.softmax(pi, dim=-1)
        return {'q': q,
                'beta': beta,
                'log_pi': log_pi,
                'pi': pi}