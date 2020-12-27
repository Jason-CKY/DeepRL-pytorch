import numpy as np
import torch.nn as nn
import torch
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from Algorithms.body import mlp, cnn, VAE

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

    def dataparallel(self, ngpu):
        print(f"Critic network using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.v_net = nn.DataParallel(self.v_net, list(range(ngpu)))

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

    def dataparallel(self, ngpu):
        print(f"Actor network using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.logits_net = nn.DataParallel(self.logits_net, list(range(ngpu)))

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

    def dataparallel(self, ngpu):
        print(f"Actor network using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.mu_net = nn.DataParallel(self.mu_net, list(range(ngpu)))

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, v_hidden_sizes=(256, 256),
                 pi_hidden_sizes=(64,64), activation=nn.Tanh, device='cpu', ngpu=1, **kwargs):
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
            self.pi_old = MLPGaussianActor(obs_dim, act_dim, pi_hidden_sizes, activation).to(device)

        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, act_dim, pi_hidden_sizes, activation).to(device)
            self.pi_old = MLPCategoricalActor(obs_dim, act_dim, pi_hidden_sizes, activation).to(device)

        self.v = MLPCritic(obs_dim, v_hidden_sizes, activation).to(device)

        self.ngpu = ngpu
        if self.ngpu > 1:
            self.pi.dataparallel(self.ngpu)
            self.pi_old.dataparallel(self.ngpu)
            self.v.dataparallel(self.ngpu)

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
    '''
    A value network for the critic of trpo
    '''
    def __init__(self, obs_dim, conv_layer_sizes, hidden_sizes, activation):
        '''
        A Multi-Layer Perceptron for the Critic network
        Args:
            obs_dim (int): observation dimension of the environment
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
        return torch.squeeze(v, -1)     # ensure v has the right shape

    def dataparallel(self, ngpu):
        print(f"Critic network using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.v_cnn = nn.DataParallel(self.v_cnn, list(range(ngpu)))
        self.v_mlp = nn.DataParallel(self.v_mlp, list(range(ngpu)))

class CNNCategoricalActor(Actor):
    '''
    Actor network for discrete outputs
    '''
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
            act_limit (float): the greatest magnitude possible for the action in the environment
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

    def calculate_kl(self, old_policy, new_policy, obs):
        """
        tf symbol for mean KL divergence between two batches of categorical probability distributions,
        where the distributions are input as log probs.
        """

        p0 = old_policy._distribution(obs).probs.detach() 
        p1 = new_policy._distribution(obs).probs

        return torch.sum(p0 * torch.log(p0 / p1), 1).mean()

    def dataparallel(self, ngpu):
        print(f"Actor network using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.logits_cnn = nn.DataParallel(self.logits_cnn, list(range(ngpu)))
        self.logits_mlp = nn.DataParallel(self.logits_mlp, list(range(ngpu)))

class CNNGaussianActor(Actor):
    '''
    Actor network for continuous outputs
    '''
    def __init__(self, obs_dim, act_dim, conv_layer_sizes, hidden_sizes, activation):
        '''
        A Convolutional Neural Net for the Actor network for Continuous outputs
        Network Architecture: (input) -> CNN -> MLP -> (output)
        Assume input is in the shape: (3, 128, 128)
        Args:
            obs_dim (tuple): observation dimension of the environment in the form of (C, H, W)
            act_dim (int): action dimension of the environment
            conv_layer_sizes (list): list of 3-tuples consisting of (output_channel, kernel_size, stride)
                                    that describes the cnn architecture
            hidden_sizes (list): list of number of neurons in each layer of MLP after output from CNN
            activation (nn.modules.activation): Activation function for each layer of MLP
            act_limit (float): the greatest magnitude possible for the action in the environment
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

    def forward_mu(self, obs):
        obs = self.mu_cnn(obs)
        obs = obs.view(-1, self.start_dim)
        mu = self.mu_mlp(obs)
        return mu

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
    
    def calculate_kl(self, old_policy, new_policy, obs):
        mu_old, std_old = old_policy.forward_mu(obs).detach(), torch.exp(old_policy.log_std).detach()
        mu, std = new_policy.forward_mu(obs), torch.exp(new_policy.log_std)

        # kl divergence between old policy and new policy : D( pi_old || pi_new )
        # (https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians)
        kl = torch.log(std/std_old) + (std_old.pow(2)+(mu_old-mu).pow(2))/(2.0*std.pow(2)) - 0.5
        return kl.sum(-1, keepdim=True).mean()

    def dataparallel(self, ngpu):
        print(f"Actor network using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.mu_cnn = nn.DataParallel(self.mu_cnn, list(range(ngpu)))
        self.mu_mlp = nn.DataParallel(self.mu_mlp, list(range(ngpu)))

class CNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, conv_layer_sizes, v_hidden_sizes=(256, 256), 
                pi_hidden_sizes=(64,64), activation=nn.Tanh, device='cpu', ngpu=1, **kwargs):
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
            self.pi_old = CNNGaussianActor(obs_dim, act_dim, conv_layer_sizes, pi_hidden_sizes, activation).to(device)

        elif isinstance(action_space, Discrete):
            self.pi = CNNCategoricalActor(obs_dim, act_dim, conv_layer_sizes, pi_hidden_sizes, activation).to(device)
            self.pi_old = CNNCategoricalActor(obs_dim, act_dim, conv_layer_sizes, pi_hidden_sizes, activation).to(device)

        self.v = CNNCritic(obs_dim, conv_layer_sizes, v_hidden_sizes, activation).to(device)

        self.ngpu = ngpu
        if self.ngpu > 1:
            self.pi.dataparallel(self.ngpu)
            self.pi_old.dataparallel(self.ngpu)
            self.v.dataparallel(self.ngpu)

    def step(self, obs):
        obs = obs.unsqueeze(0)
        self.pi.eval()
        self.v.eval()
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample().squeeze()
            # print(a.shape)
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs).detach().cpu().numpy()
        return a.detach().cpu().numpy(), v, logp_a.cpu().detach().numpy()

    def act(self, obs):
        return self.step(obs)[0]

##########################################################################################################
#VAE ACTOR-CRITIC##
##########################################################################################################
class VAECritic(nn.Module):
    def __init__(self, vae_weights_path, obs_dim, conv_layer_sizes, hidden_sizes, activation):
        '''
        A Variational Autoencoder Net for the Critic network
        Args:
            vae_weights_path (Str): Path to the vae weights file
            obs_dim (tuple): observation dimension of the environment in the form of (C, H, W)
            act_dim (int): action dimension of the environment
            hidden_sizes (list): list of number of neurons in each layer of MLP
            activation (nn.modules.activation): Activation function for each layer of MLP
        '''
        super().__init__()
        self.v_vae = VAE()
        self.v_vae.load_weights(vae_weights_path)
        self.v_mlp = mlp([self.v_vae.latent_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        '''
        Forward propagation for critic network
        Args:
            obs (Tensor [n, obs_dim]): batch of observation from environment
        '''
        obs = self.v_vae(obs)
        v = self.v_mlp(obs)
        return torch.squeeze(v, -1)     # ensure q has the right shape

    def dataparallel(self, ngpu):
        print(f"Critic network using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.v_vae.dataparallel(ngpu)
        self.v_mlp = nn.DataParallel(self.v_mlp, list(range(ngpu)))

class VAECategoricalActor(Actor):
    def __init__(self, vae_weights_path, obs_dim, act_dim, hidden_sizes, activation):
        '''
        A Variational Autoencoder Net for the Actor network for discrete outputs
        Network Architecture: (input) -> VAE -> MLP -> (output)
        Assume input is in the shape: (3, 128, 128)
        Args:
            vae_weights_path (Str): Path to the vae weights file
            obs_dim (tuple): observation dimension of the environment in the form of (C, H, W)
            act_dim (int): action dimension of the environment
            hidden_sizes (list): list of number of neurons in each layer of MLP after output from VAE
            activation (nn.modules.activation): Activation function for each layer of MLP
        '''
        super().__init__()
        
        self.logits_vae = VAE()
        self.logits_vae.load_weights(vae_weights_path)
        mlp_sizes = [self.logits_vae.latent_dim] + list(hidden_sizes) + [act_dim]
        self.logits_mlp = mlp(mlp_sizes, activation, output_activation=nn.Tanh)

        # initialise actor network final layer weights to be 1/100 of other weights
        self.logits_mlp[-2].weight.data /= 100 # last layer is Identity, so we tweak second last layer weights

    def _distribution(self, obs):
        '''
        Forward propagation for actor network
        Args:
            obs (Tensor [n, obs_dim]): batch of observation from environment
        Return:
            Categorical distribution from output of model
        '''
        obs = self.logits_vae(obs)
        logits = self.logits_mlp(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        '''
        Args:
            pi: distribution from _distribution() function
            act: log probability of selecting action act from the given distribution pi
        '''
        return pi.log_prob(act)

    def dataparallel(self, ngpu):
        print(f"Actor network using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.logits_vae.dataparallel(ngpu)
        self.logits_mlp = nn.DataParallel(self.logits_mlp, list(range(ngpu)))

class VAEGaussianActor(Actor):
    def __init__(self, vae_weights_path, obs_dim, act_dim, conv_layer_sizes, hidden_sizes, activation):
        '''
        A Convolutional Neural Net for the Actor network for Continuous outputs
        Network Architecture: (input) -> VAE -> MLP -> (output)
        Assume input is in the shape: (3, 128, 128)
        Args:
            vae_weights_path (Str): Path to the vae weights file
            obs_dim (tuple): observation dimension of the environment in the form of (C, H, W)
            act_dim (int): action dimension of the environment
            hidden_sizes (list): list of number of neurons in each layer of MLP after output from VAE
            activation (nn.modules.activation): Activation function for each layer of MLP
        '''
        super().__init__()
        log_std = -0.5*np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.mu_vae = VAE()
        mlp_sizes = [self.mu_vae.latent_dim] + list(hidden_sizes) + [act_dim]
        self.mu_mlp = mlp(mlp_sizes, activation, output_activation=nn.Tanh)
        # initialise actor network final layer weights to be 1/100 of other weights
        self.mu_mlp[-2].weight.data /= 100 # last layer is Identity, so we tweak second last layer weights

    def _distribution(self, obs):
        '''
        Forward propagation for actor network
        Args:
            obs (Tensor [n, obs_dim]): batch of observation from environment
        Return:
            Categorical distribution from output of model
        '''
        obs = self.mu_vae(obs)
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

    def dataparallel(self, ngpu):
        print(f"Actor network using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.mu_vae.dataparallel(ngpu)
        self.mu_mlp = nn.DataParallel(self.mu_mlp, list(range(ngpu)))

class VAEActorCritic(nn.Module):
    def __init__(self, vae_weights_path, observation_space, action_space, conv_layer_sizes, 
                v_hidden_sizes=(256, 256), pi_hidden_sizes=(64,64), 
                activation=nn.Tanh, device='cpu', ngpu=1, **kwargs):
        '''
        A Variational Autoencoder for the Actor_Critic network
        Args:
            vae_weights_path (Str): Path to the vae weights file
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
            self.pi = VAEGaussianActor(vae_weights_path, obs_dim, act_dim, conv_layer_sizes, pi_hidden_sizes, activation).to(device)
            self.pi_old = VAEGaussianActor(vae_weights_path, obs_dim, act_dim, conv_layer_sizes, pi_hidden_sizes, activation).to(device)

        elif isinstance(action_space, Discrete):
            self.pi = VAECategoricalActor(vae_weights_path, obs_dim, act_dim, conv_layer_sizes, pi_hidden_sizes, activation).to(device)
            self.pi_old = VAECategoricalActor(vae_weights_path, obs_dim, act_dim, conv_layer_sizes, pi_hidden_sizes, activation).to(device)

        self.v = VAECritic(vae_weights_path, obs_dim, conv_layer_sizes, v_hidden_sizes, activation).to(device)

        self.ngpu = ngpu
        if self.ngpu > 1:
            self.pi.dataparallel(self.ngpu)
            self.pi_old.dataparallel(self.ngpu)
            self.v.dataparallel(self.ngpu)

    def step(self, obs):
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