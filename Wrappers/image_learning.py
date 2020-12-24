import gym
import numpy as np
import pickle
from typing import Tuple
from gym.spaces import Box

class Image_Wrapper(gym.ObservationWrapper):
    '''
    Simple wrapper to convert state observation into the rendered image for training
    '''
    def __init__(self, env, training=True):
        super(Image_Wrapper, self).__init__(env)
        H, W, C = self.env.render('rgb_array').shape
        self.observation_space = Box(0.0, 1.0, (C, H, W), dtype=np.float32)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def observation(self, observation):
        return self.env.render('rgb_array').transpose([2, 0, 1]) / 255.0

    def save(self, fname):
        return

    # @classmethod
    def load(self, filename):
        return