import gym
import numpy as np
import pickle
from typing import Tuple

class RLBench_Wrapper(gym.ObservationWrapper):
    '''
    Observation Wrapper for the RLBench environment to only output 1 of the 
    camera views during training/testing instead of a dictionary of all camera views
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
        self.observation_space = self.observation_space[view]

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def observation(self, observation):
        return observation[self.view]
    
    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)