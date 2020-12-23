'''
body.py contains utitlity functions and classes to build neural networks. 
Contains the following:
    1) MLP
    2) CNN
    3) VAE
'''
import numpy as np
import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from pl_bolts.models.autoencoders.components import resnet18_encoder, resnet18_decoder

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

def cnn(in_channels, conv_layer_sizes, activation, batchnorm=True):
  '''
  Create a Convolutional Neural Network with given number of cnn layers
  Each convolutional layer has kernel_size=2, and stride=2, which effectively
  halves the spatial dimensions and doubles the channel size.
  Args:
    con_layer_sizes (list): list of 3-tuples consisting of 
                            (output_channel, kernel_size, stride)
    in_channels (int): incoming number of channels
    num_layers (int): number of convolutional layers needed
    activation (nn.Module.Activation): Activation function after each conv layer
    batchnorm (bool): If true, add a batchnorm2d layer after activation layer
  Returns:
    nn.Sequential module for the CNN
  '''
  layers = []
  channels = in_channels
  for i in range(len(conv_layer_sizes)):
    out_channel, kernel, stride = conv_layer_sizes[i]
    layers += [nn.Conv2d(in_channels, out_channel, kernel, stride),
               activation()]
    if batchnorm:
      layers += [nn.BatchNorm2d(out_channel)]
    
    in_channels = out_channel

  return nn.Sequential(*layers)

class VAE(nn.Module):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=128, device='cpu'):
        '''
        Identical to the VAE module in RL_VAE/vae.py, but wihtout the decoder part, for use in RL algorithms
        '''
        super().__init__()
        self.device = device
        # encoder
        self.encoder = resnet18_encoder(False, False)
        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from mu and log_var
        z = self.reparameterise(mu, log_var)
        return z

    def save_weights(self, fpath):
        print('saving checkpoint...')
        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'fc_mu': self.fc_mu.state_dict(),
            'fc_var': self.fc_var.state_dict(),
            'decoder': self.decoder.state_dict()
        }
        torch.save(checkpoint, fpath)
        print(f"checkpoint saved at {fpath}")    
    
    def load_weights(self, fpath):
        if os.path.isfile(fpath):
            checkpoint = torch.load(fpath, map_location=self.device)
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.fc_mu.load_state_dict(checkpoint['fc_mu'])
            self.fc_var.load_state_dict(checkpoint['fc_var'])

            print('checkpoint loaded at {}'.format(fpath))
        else:
            raise AssertionError(f"No weights file found at {fpath}")

    def dataparallel(self, ngpu):
        print(f"using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.encoder = nn.DataParallel(self.encoder, list(range(ngpu)))
        self.decoder = nn.DataParallel(self.decoder, list(range(ngpu)))
        self.fc_mu = nn.DataParallel(self.fc_mu, list(range(ngpu)))
        self.fc_var = nn.DataParallel(self.fc_var, list(range(ngpu)))
        