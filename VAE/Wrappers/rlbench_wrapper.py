import gym
import numpy as np
import pickle
from typing import Tuple
from gym.spaces import Box

class RLBench_Wrapper(gym.ObservationWrapper):
    '''
    Observation Wrapper for the RLBench environment to only output 1 of the 
    camera views for observations in the shape (H, W, C) for data generation
    '''
    def __init__(self, env, view):
        '''
        Args:
            view (str): Dictionary key to specify which camera view to use. 
                        RLBench observation comes in a dictionary of
                        ['state', 'left_shoulder_rgb', 'right_shoulder_rgb', 'wrist_rgb', 'front_rgb']
        '''
        super().__init__(env)
        self.view = view
        H, W, C = self.observation_space[view].shape
        self.observation_space = Box(0, 255, (H, W, C), dtype=np.uint8)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def observation(self, observation):
        return (observation[self.view] * 255).astype(np.uint8)