import numpy as np
import torch.nn as nn
import torch
from Algorithms.body import mlp, cnn, VAE

##########################################################################################################
#MLP ACTOR-CRITIC##
##########################################################################################################
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

    def dataparallel(self, ngpu):
        print(f"Actor network using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.pi = nn.DataParallel(self.pi, list(range(ngpu)))

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

    def dataparallel(self, ngpu):
        print(f"Critic network using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.q = nn.DataParallel(self.q, list(range(ngpu)))

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), 
                activation=nn.ReLU, device='cpu', ngpu=1, **kwargs):
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
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # Create Actor and Critic networks
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)
        self.q = MLPCritic(obs_dim, act_dim, hidden_sizes, activation).to(device)

        self.ngpu = ngpu
        if self.ngpu > 1:
            self.pi.dataparallel(self.ngpu)
            self.q.dataparallel(self.ngpu)
        
    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()


##########################################################################################################
#CNN ACTOR-CRITIC##
##########################################################################################################
class CNNActor(nn.Module):
    def __init__(self, obs_dim, act_dim, conv_layer_sizes, hidden_sizes, activation, act_limit):
        '''
        A Convolutional Neural Net for the Actor network
        Network Architecture: (input) -> CNN -> MLP -> (output)
        Assume observation space is in the shape: (3, 128, 128)
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
        
        self.pi_cnn = cnn(obs_dim[0], conv_layer_sizes, activation, batchnorm=True)
        self.start_dim = self.calc_shape(obs_dim, self.pi_cnn)
        mlp_sizes = [self.start_dim] + list(hidden_sizes) + [act_dim]
        self.pi_mlp = mlp(mlp_sizes, activation, output_activation=nn.Tanh)
        self.act_limit = act_limit

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
        Forward propagation for actor network
        Args:
            obs (Tensor [n, obs_dim]): batch of observation from environment
        Return:
            output of actor network * act_limit
        '''
        obs = self.pi_cnn(obs)
        obs = obs.view(-1, self.start_dim)
        obs = self.pi_mlp(obs)
        return obs*self.act_limit

    def dataparallel(self, ngpu):
        print(f"Actor Network using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.pi_cnn = nn.DataParallel(self.pi_cnn, list(range(ngpu)))
        self.pi_mlp = nn.DataParallel(self.pi_mlp, list(range(ngpu)))

class CNNCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, conv_layer_sizes, hidden_sizes, activation):
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
        self.q_cnn = cnn(obs_dim[0], conv_layer_sizes, activation, batchnorm=True)
        self.start_dim = self.calc_shape(obs_dim, self.q_cnn)
        self.q_mlp = mlp([self.start_dim + act_dim] + list(hidden_sizes) + [1], activation)

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

    def forward(self, obs, act):
        '''
        Forward propagation for critic network
        Args:
            obs (Tensor [n, obs_dim]): batch of observation from environment
            act (Tensor [n, act_dim]): batch of actions taken by actor
        '''
        obs = self.q_cnn(obs)
        obs = obs.view(-1, self.start_dim)
        q = self.q_mlp(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)     # ensure q has the right shape

    def dataparallel(self, ngpu):
        print(f"Critic Network using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.q_cnn = nn.DataParallel(self.q_cnn, list(range(ngpu)))
        self.q_mlp = nn.DataParallel(self.q_mlp, list(range(ngpu)))

class CNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, conv_layer_sizes, 
    hidden_sizes=(256, 256), activation=nn.ReLU, device='cpu', ngpu=1, **kwargs):
        '''
        A CNN Perceptron for the Actor_Critic network
        Args:
            observation_space (gym.spaces): observation space of the environment
            action_space (gym.spaces): action space of the environment
            conv_layer_sizes (list): list of 3-tuples consisting of (output_channel, kernel_size, stride)
                        that describes the cnn architecture
            hidden_sizes (tuple): list of number of neurons in each layer of MLP
            activation (nn.modules.activation): Activation function for each layer of MLP
            device (str): whether to use cpu or gpu to run the model
        '''
        super().__init__()
        obs_dim = observation_space.shape
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # Create Actor and Critic networks
        self.pi = CNNActor(obs_dim, act_dim, conv_layer_sizes, hidden_sizes, activation, act_limit).to(device)
        self.q = CNNCritic(obs_dim, act_dim, conv_layer_sizes, hidden_sizes, activation).to(device)
    
        self.ngpu = ngpu
        if self.ngpu > 1:
            self.pi.dataparallel(self.ngpu)
            self.q.dataparallel(self.ngpu)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()

##########################################################################################################
#VAE ACTOR-CRITIC##
##########################################################################################################
class VAEActor(nn.Module):
    def __init__(self, vae_weights_path, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        '''
        A Variational Autoencoder for the Actor network
        Network Architecture: (input) -> VAE -> MLP -> (output)
        The VAE is pretrained on observation images.
        Assume observation space is in the shape: (3, 128, 128)
        Args:
            vae_weights_path (Str): Path to the vae weights file
            obs_dim (tuple): observation dimension of the environment in the form of (C, H, W)
            act_dim (int): action dimension of the environment
            hidden_sizes (list): list of number of neurons in each layer of MLP after output from CNN
            activation (nn.modules.activation): Activation function for each layer of MLP
            act_limit (float): the greatest magnitude possible for the action in the environment
        '''
        super().__init__()

        self.pi_vae = VAE()
        self.pi_vae.load_weights(vae_weights_path)
        mlp_sizes = [self.pi_vae.latent_dim] + list(hidden_sizes) + [act_dim]
        self.pi_mlp = mlp(mlp_sizes, activation, output_activation=nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        '''
        Forward propagation for actor network
        Args:
            obs (Tensor [n, obs_dim]): batch of observation from environment
        Return:
            output of actor network * act_limit
        '''
        obs = self.pi_vae(obs)
        obs = self.pi_mlp(obs)
        return obs*self.act_limit
       
    def dataparallel(self, ngpu):
        print(f"Actor Network using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.pi_vae.dataparallel(ngpu)
        self.pi_mlp = nn.DataParallel(self.pi_mlp, list(range(ngpu)))

class VAECritic(nn.Module):
    def __init__(self, vae_weights_path, obs_dim, act_dim, hidden_sizes, activation):
        '''
        A Variational Autoencoder for the Critic network
        Args:
            vae_weights_path (Str): Path to the vae weights file
            obs_dim (tuple): observation dimension of the environment in the form of (C, H, W)
            act_dim (int): action dimension of the environment
            hidden_sizes (list): list of number of neurons in each layer of MLP
            activation (nn.modules.activation): Activation function for each layer of MLP
        '''
        super().__init__()

        self.q_vae = VAE()
        self.q_vae.load_weights(vae_weights_path)
        self.q_mlp = mlp([self.q_vae.latent_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        '''
        Forward propagation for critic network
        Args:
            obs (Tensor [n, obs_dim]): batch of observation from environment
            act (Tensor [n, act_dim]): batch of actions taken by actor
        '''
        obs = self.q_vae(obs)
        q = self.q_mlp(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)     # ensure q has the right shape

    def dataparallel(self, ngpu):
        print(f"Critic Network using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.q_vae.dataparallel(ngpu)
        self.q_mlp = nn.DataParallel(self.q_mlp, list(range(ngpu)))

class VAEActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, vae_weights_path,
    hidden_sizes=(256, 256), activation=nn.ReLU, device='cpu', ngpu=1, **kwargs):
        '''
        A Variational Autoencoder network for the Actor_Critic network
        Args:
            observation_space (gym.spaces): observation space of the environment
            action_space (gym.spaces): action space of the environment
            vae_weights_path (Str): Path to the vae weights file
            hidden_sizes (tuple): list of number of neurons in each layer of MLP
            activation (nn.modules.activation): Activation function for each layer of MLP
            device (str): whether to use cpu or gpu to run the model
        '''
        super().__init__()
        obs_dim = observation_space.shape
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # Create Actor and Critic networks
        self.pi = VAEActor(vae_weights_path, obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)
        self.q = VAECritic(vae_weights_path, obs_dim, act_dim, hidden_sizes, activation).to(device)
    
        self.ngpu = ngpu
        if self.ngpu > 1:
            self.pi.dataparallel(self.ngpu)
            self.q.dataparallel(self.ngpu)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()