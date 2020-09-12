"""
Adapted from:
https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/utils.py
"""

import numpy as np

class OUNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = sigma
        self.action_dim   = action_dim
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action): 
        ou_state = self.evolve_state()
        return action + ou_state