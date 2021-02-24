import pytorch_lightning as pl
import torch
import os
import torch.nn as nn
from torch.nn import functional as F
from pl_bolts.models.autoencoders.components import resnet18_encoder, resnet18_decoder
from collections import OrderedDict

class VAE(nn.Module):
    def __init__(self, beta=4, enc_out_dim=512, latent_dim=256, input_height=128, device='cpu'):
        super().__init__()

        self.beta = beta
        self.latent_dim = latent_dim
        self.device = device
        self.input_height = input_height
        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = nn.Sequential(
            resnet18_decoder(
            latent_dim=latent_dim, 
            input_height=input_height, 
            first_conv=False, 
            maxpool1=False), 
            nn.Tanh()   # Tanh activation to clamp values to [-1, 1] of the input
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        self.p = 0.2

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2, 3))

    def encode_image(self, x):
        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from mu and log_var
        z = self.reparameterise(mu, log_var)

        return z

    def get_elbo_loss(self, x, p):
        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from mu and log_var
        z = self.reparameterise(mu, log_var)
        x_hat = self.decoder(z)
        
        boot_recon = int(p * self.input_height * self.input_height * 3)
        recon_loss = F.mse_loss(x_hat, x, reduction='none').view(-1).topk(boot_recon, sorted=False)[0].sum()

        # recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        kld = 0.5*torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))
        elbo = recon_loss + self.beta * kld

        log_dict = {
            'elbo': elbo.item(),
            'recon_loss': recon_loss.item(),
            'kl': kld.item()
        }
        return elbo, log_dict

    def reconstruct(self, n_preds, sampled_noise=None):
        '''
        Decode from a normal distribution to give images
        '''

        if sampled_noise is None:
            # Z COMES FROM NORMAL(0, 1)
            p = torch.distributions.Normal(torch.zeros((self.latent_dim,)), torch.ones((self.latent_dim,)))
            z = p.rsample((n_preds,))
        else:
            z = sampled_noise

        # SAMPLE IMAGES
        with torch.no_grad():
            pred = self.decoder(z.to(self.device)).cpu()

        return pred

    def forward(self, x):
        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from mu and log_var
        z = self.reparameterise(mu, log_var)
        x_hat = self.decoder(z)

        return x_hat

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
            self.encoder.load_state_dict(self.sanitise_state_dict(checkpoint['encoder']))
            self.fc_mu.load_state_dict(self.sanitise_state_dict(checkpoint['fc_mu']))
            self.fc_var.load_state_dict(self.sanitise_state_dict(checkpoint['fc_var']))
            self.decoder.load_state_dict(self.sanitise_state_dict(checkpoint['decoder']))

            print('checkpoint loaded at {}'.format(fpath))
        else:
            raise AssertionError(f"No weights file found at {fpath}")

    def dataparallel(self, ngpu):
        print(f"using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.encoder = nn.DataParallel(self.encoder, list(range(ngpu)))
        self.decoder = nn.DataParallel(self.decoder, list(range(ngpu)))
        self.fc_mu = nn.DataParallel(self.fc_mu, list(range(ngpu)))
        self.fc_var = nn.DataParallel(self.fc_var, list(range(ngpu)))

    def sanitise_state_dict(self, state_dict):
        '''
        Weights saved with nn.DataParallel wrapper cannot be loaded with a normal net
        This utility function serves to remove the module. prefix so that the state_dict can 
        be loaded without nn.DataParallel wrapper
        Args:
            state_dict (OrderedDict): the weights to be loaded
        Returns:
            output_dict (OrderedDict): weights that is able to be loaded without nn.DataParallel wrapper
        '''
        output_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' in k:
                output_dict[k[7:]] = v # remove the first 7 characters 'module.' with string slicing
            else:
                output_dict[k] = v
        return output_dict