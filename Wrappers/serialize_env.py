import gym
import numpy as np
import pickle
from typing import Tuple

class Serialize_Env(gym.ObservationWrapper):
    '''
    Simple wrapper to add the save and load functionality
    '''
    def __init__(self, env, training=True):
        super(Serialize_Env, self).__init__(env)
    
    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)