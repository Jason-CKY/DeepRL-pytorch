import numpy as np
import torch.nn as nn
import torch
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

# need a value function and a policy function (new and old)
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

class MLPCritic(nn.Module):
    '''
    A value network for the critic of trpo
    '''
    def __init__(self, obs_dim, hidden_sizes, activation):
        '''
        A Multi-Layer Perceptron for the Critic network
        Args:
            obs_dim (int): observation dimension of the environment
            hidden_sizes (list): list of number of neurons in each layer of MLP
            activation (nn.modules.activation): Activation function for each layer of MLP
        '''
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1],  activation)
    
    def forward(self, obs):
        '''
        Forward propagation for critic network
        Args:
            obs (Tensor [n, obs_dim]): batch of observation from environment
        '''
        return torch.squeeze(self.v_net(obs), -1)     # ensure v has the right shape

class Actor(nn.Module):
    '''
    Base Actor class for categorical/gaussian actor to inherit from
    '''
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        '''
        Produce action distributions for given observations, and 
        optionally compute the log likelihood of given actions under
        those distributions
        '''
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCategoricalActor(Actor):
    '''
    Actor network for discrete outputs
    '''
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
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.logits_net[-2].weight.data /= 100 # last layer is Identity, so we tweak second last layer weights

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)
    
    def _log_prob_from_distribution(self, pi, act):
        '''
        Args:
            pi: distribution from _distribution() function
            act: log probability of selecting action act from the given distribution pi
        '''
        return pi.log_prob(act)

    def calculate_kl(self, old_policy, new_policy, obs):
        """
        tf symbol for mean KL divergence between two batches of categorical probability distributions,
        where the distributions are input as log probs.
        """

        p0 = old_policy._distribution(obs).probs.detach() 
        p1 = new_policy._distribution(obs).probs

        return torch.sum(p0 * torch.log(p0 / p1), 1).mean()

class MLPGaussianActor(Actor):
    '''
    Actor network for continuous outputs
    '''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        '''
        A Multi-Layer Perceptron for the gaussian Actor network for continuous actions
        Args:
            obs_dim (int): observation dimension of the environment
            act_dim (int): action dimension of the environment
            hidden_sizes (list): list of number of neurons in each layer of MLP
            activation (nn.modules.activation): Activation function for each layer of MLP
        '''
        super().__init__()
        log_std = -0.5*np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.mu_net[-2].weight.data /= 100 # last layer is Identity, so we tweak second last layer weights
    
    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)
    
    def _log_prob_from_distribution(self, pi, act):
        '''
        Args:
            pi: distribution from _distribution() function
            act: log probability of selecting action act from the given distribution pi
        '''
        return pi.log_prob(act).sum(axis=-1)    # last axis sum needed for Torch Normal Distribution
    
    def calculate_kl(self, old_policy, new_policy, obs):
        mu_old, std_old = old_policy.mu_net(obs).detach(), torch.exp(old_policy.log_std).detach()
        mu, std = new_policy.mu_net(obs), torch.exp(new_policy.log_std)

        # kl divergence between old policy and new policy : D( pi_old || pi_new )
        # (https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians)
        kl = torch.log(std/std_old) + (std_old.pow(2)+(mu_old-mu).pow(2))/(2.0*std.pow(2)) - 0.5
        return kl.sum(-1, keepdim=True).mean()


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, v_hidden_sizes=(256, 256), pi_hidden_sizes=(64,64), activation=nn.Tanh, device='cpu'):
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
        try:
            act_dim = action_space.shape[0]
        except IndexError:
            act_dim = action_space.n
            
        # Create Actor and Critic networks
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, act_dim, pi_hidden_sizes, activation).to(device)
            self.pi_old = MLPGaussianActor(obs_dim, act_dim, pi_hidden_sizes, activation).to(device)

        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, act_dim, pi_hidden_sizes, activation).to(device)
            self.pi_old = MLPCategoricalActor(obs_dim, act_dim, pi_hidden_sizes, activation).to(device)

        self.v = MLPCritic(obs_dim, v_hidden_sizes, activation).to(device)
    
    def step(self, obs):
        self.pi.eval()
        self.v.eval()
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs).detach().cpu().numpy()
        return a.detach().cpu().numpy(), v, logp_a.cpu().detach().numpy()

    def act(self, obs):
        return self.step(obs)[0]