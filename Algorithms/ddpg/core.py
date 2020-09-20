import numpy as np
import torch.nn as nn
import torch

def mlp(sizes, activation, output_activation=nn.Identity):
    # create a multi-layer perceptron model from input sizes and activations
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j<len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class MLPActor(nn.Module):
    '''
    A Multi-Layer Perceptron for the Actor network
    '''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, output_activation=nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # return the output scaled to action space limits
        return self.pi(obs)*self.act_limit

class MLPCritic(nn.Module):
    '''
    A Multi-Layer Perceptron for the Critic network
    '''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
    
    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)     # ensure q has the right shape

class MLPActorCritic(nn.Module):
    '''
    A Multi-Layer Perceptron for Actor and Critic networks
    '''
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU, device='cpu'):
        '''
        docstring
        '''

        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # Create Actor and Critic networks
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)
        self.q = MLPCritic(obs_dim, act_dim, hidden_sizes, activation).to(device)
    
    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()