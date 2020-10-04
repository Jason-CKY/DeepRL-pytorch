import numpy as np
import random
import torch
from copy import deepcopy

def combined_shape(length, shape=None):
    '''
    Combine the shape with batch size.
    This is to ensure that the shape will be correct if the act_dim or obs_dim
    is more than 1D array.
    '''
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    """
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    running_sum = 0
    output = deepcopy(x)
    for i in reversed(range(len(x))):
        output[i] += running_sum*discount
        running_sum = output[i]
    return output

class GAEBuffer:
    """
    A buffer for storing trajectories experienced by a TRPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, obs_dim, act_dim, size, device, gamma=0.99, lam=0.97):
        '''
        Initialise the GAE Buffer. The buffer class contains the following buffers:
        obs_buf (np.ndarray): array of observations from environment
        act_buf (np.ndarray): array of actions taken in the environment by agent
        rew_buf (np.ndarray): array of rewards obtained from the environment
        done_buf (np.ndarray): array of terminals, 1 for terminal state, 0 for non-terminal state
        ret_buf (np.ndarray): array of gamma discounted returns from the environment
        adv_buf (np.ndarray): array of advantage estimate at each timestep
        v_buf (np.ndarray): array of value estimate at each timestep
        Args:
            obs_dim: dimension size of the observation space
            act_dim: dimension size of the action space
            size (int): max size of the buffer
            device (str): use cpu/gpu to calculate
            gamma (float): discount factor for advantage estimation
            lam (float): lambda for advantage estimation
        '''
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size, f"{self.ptr}, {self.max_size}"      # Buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr
        
    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size, f"{self.ptr}"     # Buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # The next line implement the advantage normalization trick to reduce variance
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / self.adv_buf.std()
        return dict(obs=torch.Tensor(self.obs_buf).to(self.device),
                    act=torch.Tensor(self.act_buf).to(self.device),
                    ret=torch.Tensor(self.ret_buf).to(self.device),
                    adv=torch.Tensor(self.adv_buf).to(self.device),
                    logp=torch.Tensor(self.logp_buf).to(self.device))