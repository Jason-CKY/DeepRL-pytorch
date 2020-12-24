import numpy as np
import torch.nn as nn
import torch
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from Algorithms.body import mlp, cnn

##########################################################################################################
#MLP ACTOR-CRITIC##
##########################################################################################################
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
        # initialise actor network final layer weights to be 1/100 of other weights
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


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, v_hidden_sizes=(256, 256),
                 pi_hidden_sizes=(64,64), activation=nn.Tanh, device='cpu', **kwargs):
        '''
        A Multi-Layer Perceptron for the Actor_Critic network
        Args:
            observation_space (gym.spaces): observation space of the environment
            action_space (gym.spaces): action space of the environment
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

        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, act_dim, pi_hidden_sizes, activation).to(device)

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

##########################################################################################################
#CNN ACTOR-CRITIC##
##########################################################################################################
class CNNCritic(nn.Module):
    def __init__(self, obs_dim, conv_layer_sizes, hidden_sizes, activation):
        '''
        A Convolutional Neural Net for the Critic network
        Args:
            obs_dim (tuple): observation dimension of the environment in the form of (C, H, W)
            act_dim (int): action dimension of the environment
            conv_layer_sizes (list): list of 3-tuples consisting of (output_channel, kernel_size, stride)
                        that describes the cnn architecture
            hidden_sizes (list): list of number of neurons in each layer of MLP
            activation (nn.modules.activation): Activation function for each layer of MLP
        '''
        super().__init__()
        self.v_cnn = cnn(obs_dim[0], conv_layer_sizes, activation, batchnorm=True)
        self.start_dim = self.calc_shape(obs_dim, self.v_cnn)
        self.v_mlp = mlp([self.start_dim] + list(hidden_sizes) + [1], activation)

    def calc_shape(self, obs_dim, cnn):
      '''
      Function to determine the shape of the data after the conv layers
      to determine how many neurons for the MLP.
      '''
      C, H, W = obs_dim
      dummy_input = torch.randn(1, C, H, W)
      with torch.no_grad():
        cnn_out = cnn(dummy_input)
      shape = cnn_out.view(-1, ).shape[0]
      return shape

    def forward(self, obs):
        '''
        Forward propagation for critic network
        Args:
            obs (Tensor [n, obs_dim]): batch of observation from environment
        '''
        obs = self.v_cnn(obs)
        obs = obs.view(-1, self.start_dim)
        v = self.v_mlp(obs)
        return torch.squeeze(v, -1)     # ensure q has the right shape

class CNNCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, conv_layer_sizes, hidden_sizes, activation):
        '''
        A Convolutional Neural Net for the Actor network for discrete outputs
        Network Architecture: (input) -> CNN -> MLP -> (output)
        Assume input is in the shape: (3, 128, 128)
        Args:
            obs_dim (tuple): observation dimension of the environment in the form of (C, H, W)
            act_dim (int): action dimension of the environment
            conv_layer_sizes (list): list of 3-tuples consisting of (output_channel, kernel_size, stride)
                                    that describes the cnn architecture
            hidden_sizes (list): list of number of neurons in each layer of MLP after output from CNN
            activation (nn.modules.activation): Activation function for each layer of MLP
        '''
        super().__init__()
        
        self.logits_cnn = cnn(obs_dim[0], conv_layer_sizes, activation, batchnorm=True)
        self.start_dim = self.calc_shape(obs_dim, self.logits_cnn)
        mlp_sizes = [self.start_dim] + list(hidden_sizes) + [act_dim]
        self.logits_mlp = mlp(mlp_sizes, activation, output_activation=nn.Tanh)
        # initialise actor network final layer weights to be 1/100 of other weights
        self.logits_mlp[-2].weight.data /= 100 # last layer is Identity, so we tweak second last layer weights


    def calc_shape(self, obs_dim, cnn):
      '''
      Function to determine the shape of the data after the conv layers
      to determine how many neurons for the MLP.
      '''
      C, H, W = obs_dim
      dummy_input = torch.randn(1, C, H, W)
      with torch.no_grad():
        cnn_out = cnn(dummy_input)
      shape = cnn_out.view(-1, ).shape[0]
      return shape

    def _distribution(self, obs):
        '''
        Forward propagation for actor network
        Args:
            obs (Tensor [n, obs_dim]): batch of observation from environment
        Return:
            Categorical distribution from output of model
        '''
        obs = self.logits_cnn(obs)
        obs = obs.view(-1, self.start_dim)
        logits = self.logits_mlp(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        '''
        Args:
            pi: distribution from _distribution() function
            act: log probability of selecting action act from the given distribution pi
        '''
        return pi.log_prob(act)

class CNNGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, conv_layer_sizes, hidden_sizes, activation):
        '''
        A Convolutional Neural Net for the Actor network for Continuous outputs
        Network Architecture: (input) -> CNN -> MLP -> (output)
        Assume input is in the shape: (3, 128, 128)
        Args:
            obs_dim (tuple): observation dimension of the environment in the form of (C. H, W)
            act_dim (int): action dimension of the environment
            conv_layer_sizes (list): list of 3-tuples consisting of (output_channel, kernel_size, stride)
                                    that describes the cnn architecture
            hidden_sizes (list): list of number of neurons in each layer of MLP after output from CNN
            activation (nn.modules.activation): Activation function for each layer of MLP
        '''
        super().__init__()
        log_std = -0.5*np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.mu_cnn = cnn(obs_dim[0], conv_layer_sizes, activation, batchnorm=True)
        self.start_dim = self.calc_shape(obs_dim, self.mu_cnn)
        mlp_sizes = [self.start_dim] + list(hidden_sizes) + [act_dim]
        self.mu_mlp = mlp(mlp_sizes, activation, output_activation=nn.Tanh)
        # initialise actor network final layer weights to be 1/100 of other weights
        self.mu_mlp[-2].weight.data /= 100 # last layer is Identity, so we tweak second last layer weights

    def calc_shape(self, obs_dim, cnn):
      '''
      Function to determine the shape of the data after the conv layers
      to determine how many neurons for the MLP.
      '''
      C, H, W = obs_dim
      dummy_input = torch.randn(1, C, H, W)
      with torch.no_grad():
        cnn_out = cnn(dummy_input)
      shape = cnn_out.view(-1, ).shape[0]
      return shape

    def _distribution(self, obs):
        '''
        Forward propagation for actor network
        Args:
            obs (Tensor [n, obs_dim]): batch of observation from environment
        Return:
            Categorical distribution from output of model
        '''
        obs = self.mu_cnn(obs)
        obs = obs.view(-1, self.start_dim)
        mu = self.mu_mlp(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        '''
        Args:
            pi: distribution from _distribution() function
            act: log probability of selecting action act from the given distribution pi
        '''
        return pi.log_prob(act).sum(axis=-1)    # last axis sum needed for Torch Normal Distribution

class CNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, conv_layer_sizes, 
                v_hidden_sizes=(256, 256), pi_hidden_sizes=(64,64), 
                activation=nn.Tanh, device='cpu', **kwargs):
        '''
        A CNN Perceptron for the Actor_Critic network
        Args:
            observation_space (gym.spaces): observation space of the environment
            action_space (gym.spaces): action space of the environment
            conv_layer_sizes (list): list of 3-tuples consisting of (output_channel, kernel_size, stride)
                        that describes the cnn architecture
            v_hidden_sizes (tuple): list of number of neurons in each layer of MLP in value network
            pi_hidden_sizes (tuple): list of number of neurons in each layer of MLP in policy network
            activation (nn.modules.activation): Activation function for each layer of MLP
            device (str): whether to use cpu or gpu to run the model
        '''
        super().__init__()
        obs_dim = observation_space.shape
        try:
            act_dim = action_space.shape[0]
        except IndexError:
            act_dim = action_space.n
            
        # Create Actor and Critic networks
        if isinstance(action_space, Box):
            self.pi = CNNGaussianActor(obs_dim, act_dim, conv_layer_sizes, pi_hidden_sizes, activation).to(device)

        elif isinstance(action_space, Discrete):
            self.pi = CNNCategoricalActor(obs_dim, act_dim, conv_layer_sizes, pi_hidden_sizes, activation).to(device)

        self.v = CNNCritic(obs_dim, conv_layer_sizes, v_hidden_sizes, activation).to(device)
    
    def step(self, obs):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        self.pi.eval()
        self.v.eval()
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample().squeeze()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs).detach().cpu().numpy()
        return a.detach().cpu().numpy(), v, logp_a.cpu().detach().numpy()

    def act(self, obs):
        return self.step(obs)[0]