import torch
import torch.nn as nn
import numpy as np
from Algorithms.body import mlp, cnn, VAE
from gym.spaces import Box, Discrete
from torch.distributions import Categorical, Bernoulli, Normal
from torch.nn import functional as F

class MLPCategoricalActor(nn.Module):
    '''
    Actor network for discrete outputs
    '''
    def __init__(self, state_dim, act_dim, hidden_sizes, num_options, activation=nn.ReLU):
        '''
        A Multi-Layer Perceptron for the Critic network
        Args:
            state_dim (int): observation dimension of the environment
            act_dim (int): action dimension of the environment
            hidden_sizes (list): list of number of neurons in each layer of MLP
            activation (nn.modules.activation): Activation function for each layer of MLP
        '''
        super().__init__()
        self.num_options = num_options
        self.act_dim = act_dim
        self.logits_net = mlp([state_dim] + list(hidden_sizes) + [act_dim*num_options], activation)
        # initialise actor network final layer weights to be 1/100 of other weights
        self.logits_net[-2].weight.data /= 100 # last layer is Identity, so we tweak second last layer weights

    def forward(self, state, option):
        logits = self.logits_net(state).view(-1, self.num_options, self.act_dim)
        try:
            logits = logits[torch.arange(len(option)), option]
        except TypeError:
            logits = logits[:, option]

        dist = Categorical(logits=logits)
        action = dist.sample()
        logp, entropy = dist.log_prob(action), dist.entropy()

        return action.squeeze().cpu().numpy(), logp.squeeze(), entropy.squeeze()

class MLPGaussianActor(nn.Module):
    '''
    Actor network for discrete outputs
    '''
    def __init__(self, state_dim, act_dim, hidden_sizes, num_options, act_limit, activation=nn.ReLU):
        '''
        A Multi-Layer Perceptron for the Critic network
        Args:
            state_dim (int): observation dimension of the environment
            act_dim (int): action dimension of the environment
            hidden_sizes (list): list of number of neurons in each layer of MLP
            activation (nn.modules.activation): Activation function for each layer of MLP
        '''
        super().__init__()
        self.num_options = num_options
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.mu_net = mlp([state_dim] + list(hidden_sizes) + [act_dim*num_options], activation)
        self.log_std_net_net = mlp([state_dim] + list(hidden_sizes) + [act_dim*num_options], activation)
        # initialise actor network final layer weights to be 1/100 of other weights
        self.mu_net[-2].weight.data /= 100 # last layer is Identity, so we tweak second last layer weights
        self.log_std_net_net[-2].weight.data /= 100 # last layer is Identity, so we tweak second last layer weights
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state, option):
        mu = self.mu_net(state).view(-1, self.num_options, self.act_dim)
        log_std = self.log_std_net_net(state).view(-1, self.num_options, self.act_dim)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        try:
            mu = mu[torch.arange(len(option)), option]
            std = std[torch.arange(len(option)), option]
        except TypeError:
            mu = mu[:, option]
            std = std[:, option]

        if self.training:
            dist = Normal(mu, std)
            action = dist.rsample()
            # compute logp and do Tanh squashing
            logp = dist.log_prob(action).sum(axis=-1)
            logp -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1)
            entropy = dist.entropy()
        else:
            action = mu
            logp, entropy = None, None

        action = torch.tanh(action)
        action = self.act_limit * action
        return action.squeeze().detach().cpu().numpy(), logp.squeeze(), entropy.squeeze()

class OptionCriticFeatures(nn.Module):
    '''
    Option-Critic Framework, using shared dense layers as the encoder.
    '''
    def __init__(self, observation_space, action_space, num_options,
    hidden_sizes=(256,), activation=nn.ReLU, device='cpu', ngpu=1, **kwargs):
        super().__init__()
        self.action_space = action_space
        obs_dim = observation_space.shape[0]
        self.num_options = num_options
        if isinstance(self.action_space, Box):
            self.act_dim = action_space.shape[0]
            self.act_limit = action_space.high[0]
        else:
            self.act_dim = action_space.n

        self.encoder = mlp([obs_dim] + list(hidden_sizes), activation).to(device)
        # self.Q is the state-option value function, and the policy over option is chosen over the highest Q value
        self.Q = mlp([ hidden_sizes[-1] ] + [num_options], activation).to(device)
        self.terminations = mlp([ hidden_sizes[-1] ] + [num_options], activation, output_activation=nn.Sigmoid).to(device)
        # self.pi = mlp([ hidden_sizes[-1] ]+[self.num_options*self.act_dim], activation).to(device)
        if isinstance(self.action_space, Box):
            self.pi = MLPGaussianActor(hidden_sizes[-1], self.act_dim, [], num_options, self.act_limit, activation).to(device)

        elif isinstance(self.action_space, Discrete):
            self.pi = MLPCategoricalActor(hidden_sizes[-1], self.act_dim, [], num_options, activation).to(device)

        self.to(device)
        self.device = device

        self.ngpu = ngpu
        if self.ngpu > 1:
            self.encoder = nn.DataParallel(self.encoder, list(range(ngpu)))
            self.Q = nn.DataParallel(self.Q, list(range(ngpu)))
            self.terminations = nn.DataParallel(self.terminations, list(range(ngpu)))
            self.pi = nn.DataParallel(self.pi, list(range(ngpu)))

    def encode_state(self, obs):
        '''
        Encode the image observation through the VAE encoder to get feature representation
        Args:
            obs (torch.Tensor): raw pixel input from environment
        Returns:
            state (torch.Tensor): output of pre-trained VAE
        '''
        obs = obs.to(self.device)
        state = self.encoder(obs)
        return state
        
    def get_option(self, obs, eps, greedy=False):
        '''
        Use the policy over option to select option based on given input observation.
        Epsilon-greedy policy is used to select option.
        Args:
            obs (torch.Tensor): given input observation
            eps (float): epsilon value to be used in the epsilon greedy policy over options
            greedy (bool): if True, return the greedy option
        Return:
            option (int): return the option given by the policy over option
        '''
        with torch.no_grad():
            obs = obs.to(self.device)
            state = self.encode_state(obs)
            greedy_option = self.Q(state).argmax(dim=-1).item()

        if greedy:
            return greedy_option
        else:
            return np.random.choice(self.num_options) if np.random.rand() < eps else greedy_option

    def predict_option_termination(self, obs, current_option):
        '''
        Pass current observation and option to the termination network and use a Bernoulli Distributin sample
        to test if option terminates.
        Args:
            obs (torch.Tensor): Given input observation
            current_option (int): the current option used
        '''
        with torch.no_grad():
            obs = obs.to(self.device)
            state = self.encode_state(obs)
            terminations = self.terminations(state)[current_option]
            option_termination = Bernoulli(probs=terminations).sample()

        return bool(option_termination.item())

    def get_action(self, obs, current_option):
        '''
        Use option actor network to predict action given current observation.
        Args:
            obs (toch.Tensor): Given input observation
            current_option (int): the current option actor to use
        Return:
            action (numpy ndarray): Action to take
        '''
        obs = obs.to(self.device)
        state = self.encode_state(obs)
        action, logp, entropy = self.pi(state, current_option)
        if action.ndim == 0 and not isinstance(self.action_space, Discrete):
            action = np.expand_dims(action, 0)
        return action, logp, entropy

    def get_terminations(self, states):
        '''
        Pass current observation and option to the termination network
        Args:
            obs (torch.Tensor): Given input observation
        '''
        terminations = self.terminations(states)
        return terminations

    def get_Q(self, states):
        '''
        Pass current observation and option to the termination network
        Args:
            obs (torch.Tensor): Given input observation
        '''
        Q = self.Q(states)
        return Q

class OptionCriticVAE(nn.Module):
    '''
    Option-Critic Framework, using shared dense layers as the encoder.
    '''
    def __init__(self, observation_space, action_space, num_options,
    hidden_sizes=(256,), activation=nn.ReLU, device='cpu', ngpu=1, vae_weights_path=None, **kwargs):
        super().__init__()
        self.action_space = action_space
        self.num_options = num_options
        if isinstance(self.action_space, Box):
            self.act_dim = action_space.shape[0]
            self.act_limit = action_space.high[0]
        else:
            self.act_dim = action_space.n

        self.encoder = VAE().to(device)
        if vae_weights_path is not None:
            self.encoder.load_weights(vae_weights_path)
        # self.Q is the state-option value function, and the policy over option is chosen over the highest Q value
        self.Q = mlp([ self.encoder.latent_dim ] + [num_options], activation).to(device)
        self.terminations = mlp([ self.encoder.latent_dim ] + [num_options], activation, output_activation=nn.Sigmoid).to(device)
        if isinstance(self.action_space, Box):
            self.pi = MLPGaussianActor(self.encoder.latent_dim, self.act_dim, hidden_sizes, num_options, self.act_limit, activation).to(device)

        elif isinstance(self.action_space, Discrete):
            self.pi = MLPCategoricalActor(self.encoder.latent_dim, self.act_dim, hidden_sizes, num_options, activation).to(device)

        self.to(device)
        self.device = device

        self.ngpu = ngpu
        if self.ngpu > 1:
            self.encoder = self.encoder.dataparallel(ngpu)
            self.Q = nn.DataParallel(self.Q, list(range(ngpu)))
            self.terminations = nn.DataParallel(self.terminations, list(range(ngpu)))
            self.pi = nn.DataParallel(self.pi, list(range(ngpu)))

    def encode_state(self, obs):
        '''
        Encode the image observation through the VAE encoder to get feature representation
        Args:
            obs (torch.Tensor): raw pixel input from environment
        Returns:
            state (torch.Tensor): output of pre-trained VAE
        '''
        obs = obs.to(self.device)
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        state = self.encoder(obs)
        return state
        
    def get_option(self, obs, eps, greedy=False):
        '''
        Use the policy over option to select option based on given input observation.
        Epsilon-greedy policy is used to select option.
        Args:
            obs (torch.Tensor): given input observation
            eps (float): epsilon value to be used in the epsilon greedy policy over options
            greedy (bool): if True, return the greedy option
        Return:
            option (int): return the option given by the policy over option
        '''
        with torch.no_grad():
            obs = obs.to(self.device)
            state = self.encode_state(obs)
            greedy_option = self.Q(state).argmax(dim=-1).item()

        if greedy:
            return greedy_option
        else:
            return np.random.choice(self.num_options) if np.random.rand() < eps else greedy_option

    def predict_option_termination(self, obs, current_option):
        '''
        Pass current observation and option to the termination network and use a Bernoulli Distributin sample
        to test if option terminates.
        Args:
            obs (torch.Tensor): Given input observation
            current_option (int): the current option used
        '''
        with torch.no_grad():
            obs = obs.to(self.device)
            state = self.encode_state(obs)
            terminations = self.terminations(state).squeeze()
            # print(terminations.shape)
            terminations = terminations[current_option]
            option_termination = Bernoulli(probs=terminations).sample()

        return bool(option_termination.item())

    def get_action(self, obs, current_option):
        '''
        Use option actor network to predict action given current observation.
        Args:
            obs (toch.Tensor): Given input observation
            current_option (int): the current option actor to use
        Return:
            action (numpy ndarray): Action to take
        '''
        obs = obs.to(self.device)
        state = self.encode_state(obs)
        action, logp, entropy = self.pi(state, current_option)
        if action.ndim == 0 and not isinstance(self.action_space, Discrete):
            action = np.expand_dims(action, 0)
        return action, logp, entropy

    def get_terminations(self, states):
        '''
        Pass current observation and option to the termination network
        Args:
            obs (torch.Tensor): Given input observation
        '''
        terminations = self.terminations(states)
        return terminations

    def get_Q(self, states):
        '''
        Pass current observation and option to the termination network
        Args:
            obs (torch.Tensor): Given input observation
        '''
        Q = self.Q(states)
        return Q
