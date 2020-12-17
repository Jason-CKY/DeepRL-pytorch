import gym
import numpy as np
import pickle
from typing import Tuple
from gym.spaces import Box

class RLBench_Wrapper(gym.ObservationWrapper):
    '''
    Observation Wrapper for the RLBench environment to only output 1 of the 
    camera views during training/testing instead of a dictionary of all camera views
    Observation space is in the shape (128, 128, 3), while actual observations are tweaked to be
    of the shape (3, 128, 128) for ease of conversion into tensor
    '''
    def __init__(self, env, view):
        '''
        Args:
            view (str): Dictionary key to specify which camera view to use. 
                        RLBench observation comes in a dictionary of
                        ['state', 'left_shoulder_rgb', 'right_shoulder_rgb', 'wrist_rgb', 'front_rgb']
        '''
        super(RLBench_Wrapper, self).__init__(env)
        self.view = view
        if len(self.observation_space[view].shape) == 3:
            # swap (128, 128, 3) into (3, 128, 128) for torch input
            H, W, C = self.observation_space[view].shape
            self.observation_space = Box(0.0, 1.0, (C, H, W), dtype=np.float32)
        else:
            self.observation_space = self.observation_space[view]

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def observation(self, observation):
        return observation[self.view].transpose([2, 0, 1])
    
    def save(self, fname):
        return

    # @classmethod
    def load(self, filename):
        return