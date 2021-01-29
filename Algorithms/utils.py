#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def get_actor_critic_module(ac_kwargs, RL_Algorithm):
    if ac_kwargs['model_type'].lower() == 'mlp':
        if RL_Algorithm.lower() == 'ddpg':
            from Algorithms.ddpg.core import MLPActorCritic            
            return MLPActorCritic
        elif RL_Algorithm.lower() == 'td3':
            from Algorithms.td3.core import MLPActorCritic            
            return MLPActorCritic
        elif RL_Algorithm.lower() == 'trpo':
            from Algorithms.trpo.core import MLPActorCritic            
            return MLPActorCritic
        elif RL_Algorithm.lower() == 'ppo':
            from Algorithms.ppo.core import MLPActorCritic            
            return MLPActorCritic
        elif RL_Algorithm.lower() == 'option_critic':
            from Algorithms.option_critic.core import OptionCriticFeatures
            return OptionCriticFeatures

    elif ac_kwargs['model_type'].lower() == 'cnn':
        if RL_Algorithm.lower() == 'ddpg':
            from Algorithms.ddpg.core import CNNActorCritic            
            return CNNActorCritic
        elif RL_Algorithm.lower() == 'td3':
            from Algorithms.td3.core import CNNActorCritic            
            return CNNActorCritic
        elif RL_Algorithm.lower() == 'trpo':
            from Algorithms.trpo.core import CNNActorCritic            
            return CNNActorCritic
        elif RL_Algorithm.lower() == 'ppo':
            from Algorithms.ppo.core import CNNActorCritic            
            return CNNActorCritic

    elif ac_kwargs['model_type'].lower() == 'vae':
        if RL_Algorithm.lower() == 'ddpg':
            from Algorithms.ddpg.core import VAEActorCritic            
            return VAEActorCritic
        elif RL_Algorithm.lower() == 'td3':
            from Algorithms.td3.core import VAEActorCritic            
            return VAEActorCritic
        elif RL_Algorithm.lower() == 'trpo':
            from Algorithms.trpo.core import VAEActorCritic            
            return VAEActorCritic
        elif RL_Algorithm.lower() == 'ppo':
            from Algorithms.ppo.core import VAEActorCritic            
            return VAEActorCritic
        elif RL_Algorithm.lower() == 'option_critic':
            from Algorithms.option_critic.core import OptionCriticVAE
            return OptionCriticVAE
    
    raise AssertionError("Invalid model_type in config.json. Choose among ['mlp', 'cnn', 'vae']")

def sanitise_state_dict(state_dict, multi_gpu=False):
    '''
    Weights saved with nn.DataParallel wrapper cannot be loaded with a normal net
    This utility function serves to remove the module. prefix so that the state_dict can 
    be loaded without nn.DataParallel wrapper
    Args:
        state_dict (OrderedDict): the weights to be loaded
    Returns:
        output_dict (OrderedDict): weights that is able to be loaded without nn.DataParallel wrapper
    '''
    if multi_gpu:
        return state_dict
        
    output_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            output_dict[k[7:]] = v # remove the first 7 characters 'module.' with string slicing
        else:
            output_dict[k] = v
    return output_dict

def to_tensor(obs):
    '''
    Convert observation into a pytorch tensor
    '''
    if isinstance(obs, torch.Tensor):
        return obs
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs

def to_np(t):
    return t.cpu().detach().numpy()

def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]

class ConstantSchedule:
    def __init__(self, val):
        self.val = val

    def __call__(self, steps=1):
        return self.val


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val
