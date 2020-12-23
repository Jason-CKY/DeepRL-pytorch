import gym
import numpy as np
import pickle
from typing import Tuple
from gym.spaces import Box

class Image_Wrapper(gym.ObservationWrapper):
    '''
    Simple wrapper to convert state observation into the rendered image to collect images data
    '''
    def __init__(self, env, training=True):
        super().__init__(env)
        H, W, C = self.env.render('rgb_array').shape
        self.observation_space = Box(0, 255, (H, W, C), dtype=np.uint8)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def observation(self, observation):
        return self.env.render('rgb_array')