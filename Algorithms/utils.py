import numpy as np
import torch
from collections import OrderedDict
from Algorithms.ddpg.core import MLPActorCritic, CNNActorCritic

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
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs