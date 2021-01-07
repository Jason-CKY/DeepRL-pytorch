import torch
import torch.nn as nn
import numpy as np
from Algorithms.body import mlp, cnn, VAE
from torch.distributions import Categorical, Bernoulli, Normal
from torch.nn import functional as F

class Option_Actor(nn.Module):
    def __init__(self, state_dim, hidden_sizes, act_dim, num_options, 
                activation, act_limit, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.act_limit = act_limit
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_options = num_options
        self.act_dim = act_dim

        self.fc_mu = mlp([state_dim] + list(hidden_sizes) + [num_options*act_dim], activation)
        self.fc_std = mlp([state_dim] + list(hidden_sizes) + [num_options*act_dim], activation)
    
    def forward(self, states, options):
        batch_idx = torch.arange(len(options))
        mu, log_std = self.fc_mu(states), self.fc_std(states)
        mu = mu.view(-1, self.num_options, self.act_dim)
        log_std = log_std.view(-1, self.num_options, self.act_dim)

        mu = mu[batch_idx, options, :]
        log_std = log_std[batch_idx, options, :]
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        
        if self.training:
            pi_action = pi_distribution.rsample()
        else:
            pi_action = mu

        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        # correction for tanh squashing
        logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi              

class OptionCriticVAE(nn.Module):
    '''
    Option-Critic Framework, using pre-trained VAE as the encoder, and a SAC updates for value function and 
    intra-option policy.

    Note that many repos online uses a stochastic policy and correspondingly using on-policy gradient methods.
    '''
    def __init__(self, observation_space, action_space, num_options, vae_weights_path,
    hidden_sizes=(256,), activation=nn.ReLU, device='cpu', ngpu=1, **kwargs):
        super().__init__()
        obs_dim = observation_space.shape
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        self.encoder = VAE().to(device)
        # self.encoder.load_weights(vae_weights_path)
        # self.Q is the state-option value function, and the policy over option is chosen over the highest Q value
        self.Q1 = mlp([self.encoder.latent_dim] + list(hidden_sizes) + [num_options], activation).to(device)
        self.Q2 = mlp([self.encoder.latent_dim] + list(hidden_sizes) + [num_options], activation).to(device)
        self.terminations = mlp([self.encoder.latent_dim] + list(hidden_sizes) + [num_options], 
                                activation, output_activation=nn.Sigmoid).to(device)
        self.policies = Option_Actor(self.encoder.latent_dim, hidden_sizes, act_dim, num_options, 
                        activation, act_limit).to(device)

        self.to(device)
        self.num_options = num_options
        self.device = device

        self.ngpu = ngpu
        if self.ngpu > 1:
            self.encoder.dataparallel(self.ngpu)
            self.Q1 = nn.DataParallel(self.Q1, list(range(ngpu)))
            self.Q2 = nn.DataParallel(self.Q2, list(range(ngpu)))
            self.terminations = nn.DataParallel(self.terminations, list(range(ngpu)))
            self.policies = nn.DataParallel(self.policies, list(range(ngpu)))

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
            q1 = self.Q1(state).max(dim=-1).values
            q2 = self.Q2(state).max(dim=-1).values
            if q1 < q2:
                greedy_option = self.Q1(state).argmax(dim=-1).item()
            else:
                greedy_option = self.Q2(state).argmax(dim=-1).item()

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
            terminations = self.terminations(state)[:, current_option]
            option_termination = Bernoulli(probs=terminations).sample()

        return bool(option_termination.item())

    def get_terminations(self, obs):
        '''
        Pass current observation and option to the termination network
        Args:
            obs (torch.Tensor): Given input observation
        '''
        obs = obs.to(self.device)
        state = self.encode_state(obs)
        terminations = self.terminations(state)

        return terminations

    def get_action(self, obs, current_option):
        '''
        Use option actor network to predict action given current observation.
        Args:
            obs (toch.Tensor): Given input observation
            current_option (int): the current option actor to use
        Return:
            action (numpy ndarray): Action to take
        '''
        with torch.no_grad():
            obs = obs.to(self.device)
            state = self.encode_state(obs)
            action, _ = self.policies(state, [current_option])

        return action.squeeze().cpu().numpy()