import torch
import torch.nn as nn
import numpy as np
from Algorithms.utils import to_tensor
from Algorithms.body import mlp, cnn, VAE
from torch.distributions import Categorical, Bernoulli, Normal
from torch.nn import functional as F

class Option_Selection_Policy(nn.Module):
    def __init__(self, state_dim, hidden_sizes, num_options, activation):
        super().__init__()
        self.num_options = num_options
        self.fc = mlp([state_dim + num_options] + list(hidden_sizes) + [num_options], activation)
    
    def forward(self, states, prev_options):
        prev_options = F.one_hot(prev_options, num_classes=self.num_options)
        x = self.fc(torch.cat([states, prev_options], dim=-1))
        return F.softmax(x, dim=-1)

class Intra_Option_Policy(nn.Module):
    def __init__(self, state_dim, hidden_sizes, act_dim, num_options, 
                activation, act_limit, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.act_limit = act_limit
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_options = num_options
        self.act_dim = act_dim

        self.fc_mu = mlp([state_dim + num_options] + list(hidden_sizes) + [act_dim], activation)
        self.fc_std = mlp([state_dim + num_options] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, states, options, deterministic=False):
        options = F.one_hot(options, num_classes=self.num_options)
        mu = self.fc_mu(torch.cat([states, options], dim=-1))
        log_std = self.fc_std(torch.cat([states, options], dim=-1))
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        # print(mu, std)
        # print(logp_pi)
        # correction for tanh squashing
        logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi                  

class U_Approximator(nn.Module):
    def __init__(self, state_dim, hidden_sizes, num_options, activation):
        super().__init__()
        self.num_options = num_options
        self.fc = mlp([state_dim + num_options] + list(hidden_sizes) + [1], activation)

    def forward(self, states, options):
        options = F.one_hot(options, num_classes=self.num_options)
        return self.fc(torch.cat([states, options], dim=-1))

class Q_Approximator(nn.Module):
    def __init__(self, state_dim, hidden_sizes, act_dim, num_options, activation):
        super().__init__()
        self.num_options = num_options
        self.fc = mlp([state_dim + act_dim] + list(hidden_sizes) + [num_options], activation)

    def forward(self, states, options, actions):
        batch_idx = torch.arange(len(options))
        q_values = self.fc(torch.cat([states, actions], dim=-1))
        return q_values[batch_idx, options]
    

class OptionCriticVAE(nn.Module):
    '''
    Option-Critic Framework, using pre-trained VAE as the encoder, and a SAC updates for value function and 
    intra-option policy.

    Note that many repos online uses a stochastic policy and correspondingly using on-policy gradient methods.
    '''
    def __init__(self, observation_space, action_space, num_options, vae_weights_path=None,
    hidden_sizes=(256,), activation=nn.ReLU, device='cpu', ngpu=1, **kwargs):
        super().__init__()
        obs_dim = observation_space.shape
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        self.encoder = VAE().to(device)
        if vae_weights_path is not None:
            self.encoder.load_weights(vae_weights_path)

        # Actors
        self.pi_high = Option_Selection_Policy(self.encoder.latent_dim, hidden_sizes, num_options, activation).to(device)
        self.pi_low = Intra_Option_Policy(self.encoder.latent_dim, hidden_sizes, act_dim, num_options, 
                        activation, act_limit).to(device)

        # Critics
        self.U1 = U_Approximator(self.encoder.latent_dim, hidden_sizes, num_options, activation).to(device)
        self.U2 = U_Approximator(self.encoder.latent_dim, hidden_sizes, num_options, activation).to(device)

        self.Q1 = Q_Approximator(self.encoder.latent_dim, hidden_sizes, act_dim, num_options, activation).to(device)
        self.Q2 = Q_Approximator(self.encoder.latent_dim, hidden_sizes, act_dim, num_options, activation).to(device)
        

        self.to(device)
        self.num_options = num_options
        self.device = device
        self.act_limit = act_limit

        self.ngpu = ngpu
        if self.ngpu > 1:
            self.encoder.dataparallel(self.ngpu)
            self.pi_high = nn.DataParallel(self.pi_high, list(range(ngpu)))
            self.pi_low = nn.DataParallel(self.pi_low, list(range(ngpu)))
            self.Q1 = nn.DataParallel(self.Q1, list(range(ngpu)))
            self.Q2 = nn.DataParallel(self.Q2, list(range(ngpu)))
            self.U1 = nn.DataParallel(self.U1, list(range(ngpu)))
            self.U2 = nn.DataParallel(self.U2, list(range(ngpu)))


    def get_Q1(self, obs, options, actions):
        obs = obs.to(self.device)
        if not isinstance(options, torch.Tensor):
            options = torch.tensor([options], dtype=torch.long).to(self.device)
        states = self.encode_state(obs)
        return self.Q1(states, options, actions)

    def get_Q2(self, obs, options, actions):
        obs = obs.to(self.device)
        if not isinstance(options, torch.Tensor):
            options = torch.tensor([options], dtype=torch.long).to(self.device)
        states = self.encode_state(obs)
        return self.Q2(states, options, actions)

    def get_U1(self, obs, options):
        obs = obs.to(self.device)
        if not isinstance(options, torch.Tensor):
            options = torch.tensor([options], dtype=torch.long).to(self.device)
        states = self.encode_state(obs)
        return self.U1(states, options)

    def get_U2(self, obs, options):
        obs = obs.to(self.device)
        if not isinstance(options, torch.Tensor):
            options = torch.tensor([options], dtype=torch.long).to(self.device)
        states = self.encode_state(obs)
        return self.U2(states, options)

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

    def act(self, obs, option, deterministic=False):
        '''
        Use option actor network to predict action given current observation.
        Args:
            obs (toch.Tensor): Given input observation
            current_option (int): the current option actor to use
        Return:
            action (numpy ndarray): Action to take
            logp (float tensor): log probability of taking that action
        '''
        obs = obs.to(self.device)
        states = self.encode_state(obs)
        if not isinstance(option, torch.Tensor):
            option = torch.tensor([option], dtype=torch.long).to(self.device)
        actions, logp = self.pi_low(states, option, deterministic)
        actions = actions.squeeze().detach().cpu().numpy()
        return actions, logp

    def get_action_logp(self, obs, options, actions):
        '''
        get the log probability of taking said action
        '''
        obs = obs.to(self.device)
        states = self.encode_state(obs)
        if not isinstance(options, torch.Tensor):
            options = torch.tensor([options], dtype=torch.long).to(self.device)

        options = F.one_hot(options, num_classes=self.num_options)
        mu = self.pi_low.fc_mu(torch.cat([states, options], dim=-1))
        log_std = self.pi_low.fc_std(torch.cat([states, options], dim=-1))
        log_std = torch.clamp(log_std, self.pi_low.log_std_min, self.pi_low.log_std_max)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        
        # correction for the actions taken by reversing the operations from the output of pi_low
        pi_action = actions / self.act_limit
        pi_action = torch.atanh(pi_action)

        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        # correction for tanh squashing
        logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)

        return logp_pi
                
    def get_option(self, obs, prev_option, deterministic=False):
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
        states = self.encode_state(obs)
        prev_option = torch.tensor([prev_option], dtype=torch.long).to(self.device)
        option_probabilities = self.pi_high(states, prev_option)
        if deterministic:
            return option_probabilities.argmax(dim=-1).item()     
        else:
            return Categorical(probs=option_probabilities).sample().item()