import torch
import torch.nn as nn
import numpy as np
from Algorithms.body import mlp, cnn, VAE
from torch.distributions import Categorical, Bernoulli, Normal
from torch.nn import functional as F

class Intra_Option_Policy(nn.Module):
    def __init__(self, state_dim, hidden_sizes, act_dim, activation, act_limit, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.act_limit = act_limit
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc_mu = mlp([state_dim] + list(hidden_sizes) + [act_dim], activation)
        self.fc_std = mlp([state_dim] + list(hidden_sizes) + [act_dim], activation)
    
    def forward(self, state):
        mu, log_std = self.fc_mu(state), self.fc_std(state)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
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
        return pi_action, logp_pi, pi_distribution.entropy().mean()

class OptionCriticVAE(nn.Module):
    '''
    Option-Critic Framework, using pre-trained VAE as the encoder. Deterministic policies are used for
    each option network, with policy loss used in off-policy methods (namely DDPG).

    Note that many repos online uses a stochastic policy and correspondingly using on-policy gradient methods.
    '''
    def __init__(self, observation_space, action_space, num_options, vae_weights_path,
    hidden_sizes=(256,), activation=nn.ReLU, device='cpu', **kwargs):
        super().__init__()
        obs_dim = observation_space.shape
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        self.encoder = VAE().to(device)
        # self.encoder.load_weights(vae_weights_path)
        # self.Q_omega is the state-option value function, and the policy over option is chosen over the highest Q value
        self.Q_omega = mlp([self.encoder.latent_dim] + list(hidden_sizes) + [num_options], activation).to(device)
        self.terminations = mlp([self.encoder.latent_dim] + list(hidden_sizes) + [num_options], 
                                activation, output_activation=nn.Sigmoid).to(device)
        self.policies = []
        for i in range(num_options):
            self.policies.append(Intra_Option_Policy(self.encoder.latent_dim, hidden_sizes, act_dim, 
                            activation, act_limit).to(device))

        self.to(device)
        self.num_options = num_options
        self.device = device

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
        obs = obs.to(self.device)
        state = self.encode_state(obs)
        greedy_option = self.Q_omega(state).argmax(dim=-1).item()
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
            action (int): Action to take
        '''
        obs = obs.to(self.device)
        state = self.encode_state(obs)
        action, logp, entropy = self.policies[current_option](state)

        return action, logp, entropy

    def get_Q_omega(self, obs, option):
        obs = obs.to(self.device)
        state = self.encode_state(obs)
        q_omega = self.Q_omega(state)[:, option]
        return q_omega

    def get_V_omega(self, obs):
        obs = obs.to(self.device)
        state = self.encode_state(obs)
        return self.Q_omega(state).max(dim=-1).values
