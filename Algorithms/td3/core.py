import numpy as np
import torch.nn as nn
import torch

def mlp(sizes, activation, output_activation=nn.Identity):
    '''
    Create a multi-layer perceptron model from input sizes and activations
    Args:
        sizes (list): list of number of neurons in each layer of MLP
        activation (nn.modules.activation): Activation function for each layer of MLP
        output_activation (nn.modules.activation): Activation function for the output of the last layer
    Return:
        nn.Sequential module for the MLP
    '''
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j<len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        '''
        A Multi-Layer Perceptron for the Actor network
        Args:
            obs_dim (int): observation dimension of the environment
            act_dim (int): action dimension of the environment
            hidden_sizes (list): list of number of neurons in each layer of MLP
            activation (nn.modules.activation): Activation function for each layer of MLP
            act_limit (float): the greatest magnitude possible for the action in the environment
        '''
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, output_activation=nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        '''
        Forward propagation for actor network
        Args:
            obs (Tensor [n, obs_dim]): batch of observation from environment
        Return:
            output of actor network * act_limit
        '''
        return self.pi(obs)*self.act_limit

class MLPCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        '''
        A Multi-Layer Perceptron for the Critic network
        Args:
            obs_dim (int): observation dimension of the environment
            act_dim (int): action dimension of the environment
            hidden_sizes (list): list of number of neurons in each layer of MLP
            activation (nn.modules.activation): Activation function for each layer of MLP
        '''
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
    
    def forward(self, obs, act):
        '''
        Forward propagation for critic network
        Args:
            obs (Tensor [n, obs_dim]): batch of observation from environment
            act (Tensor [n, act_dim]): batch of actions taken by actor
        '''
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)     # ensure q has the right shape

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU, device='cpu'):
        '''
        A Multi-Layer Perceptron for the Actor_Critic network
        Args:
            observation_space (gym.spaces): observation space of the environment
            act_space (gym.spaces): action space of the environment
            hidden_sizes (tuple): list of number of neurons in each layer of MLP
            activation (nn.modules.activation): Activation function for each layer of MLP
            device (str): whether to use cpu or gpu to run the model
        '''
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # Create Actor and Critic networks
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)
        self.q1 = MLPCritic(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.q2 = MLPCritic(obs_dim, act_dim, hidden_sizes, activation).to(device)
    
    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()