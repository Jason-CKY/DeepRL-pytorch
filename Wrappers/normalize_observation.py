import gym
import numpy as np
import pickle
import json
from typing import Tuple

class Running_Stat:
    '''
    Class to store variables required to compute 1st and 2nd order statistics
    Adapted from: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
                  https://datagenetics.com/blog/november22017/index.html

    Methods in this class do not require a running list of sample points to compute statistics
    '''
    def __init__(self, shape: Tuple[int, ...]):
        '''
        Args:
            shape (Tuple): shape of each observation
        '''
        self.n = 0          # n is the number of observations so far

        self.mean = np.zeros(shape)
        self.M2 = np.zeros(shape)

    def update(self, x:np.ndarray):
        '''
        Adding a new observation to update the running stats
        Args:
            x (np.ndarray): observation
        '''
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self):
        return self.mean
    
    def get_svariance(self): # sample variance
        if self.n < 2:
            return np.ones(self.M2.shape)

        return self.M2 / (self.n - 1)

    def get_pvariance(self): # popn variance
        if self.n < 2:
            return np.ones(self.M2.shape)

        return self.M2 / (self.n)

class Normalize_Observation(gym.ObservationWrapper):
    '''
    # https://arxiv.org/pdf/2006.05990.pdf
    Observation normalization: 
    If enabled, we keep the empirical mean oμ and standard deviation oρ
    of each observation coordinate (based on all observations seen so far) 
    and normalize observations by subtracting the empirical mean and 
    dividing by max(oρ,10−6). This results in all neural networks inputs
    having approximately zero mean and standard deviation equal to one.  
    '''
    def __init__(self, env, training=True):
        super(Normalize_Observation, self).__init__(env)
        self.training=training
        self.eps = np.ones(self.env.observation_space.shape) * 1e-6
        self.running_stats = Running_Stat(self.env.observation_space.shape)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def observation(self, observation):
        if self.training:
            self.running_stats.update(observation)
        mean = self.running_stats.get_mean()
        var = self.running_stats.get_svariance()

        std = np.sqrt(var)
        output_observation = (observation - mean) / np.maximum(std, self.eps)
        # print(f"Mean: {output_observation.mean(axis=0)}, Std: {output_observation.std(axis=0)}")
        return output_observation
    
    def save(self, fname):
        # with open(fname, 'wb') as f:
        #     pickle.dump(self, f)
        # Cannot pickle rlbench envs as they are thread.lock objects, so save the params as json instead
        stats = {
            "n": self.running_stats.n,
            "mean": self.running_stats.mean.tolist(),
            "M2": self.running_stats.M2.tolist()
        }
        with open(fname, 'w') as f:
            f.write(json.dumps(stats, indent=4))

    # @classmethod
    def load(self, filename):
        # with open(filename, 'rb') as f:
        #     return pickle.load(f)
        # Cannot pickle rlbench envs as they are thread.lock objects, so load the params as json instead
        with open(filename, 'r') as f:
            stats = json.load(f)
        self.running_stats.n = stats['n']
        self.running_stats.mean = np.array(stats['mean'])
        self.running_stats.M2 = np.array(stats['M2'])