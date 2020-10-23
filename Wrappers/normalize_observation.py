import gym
import numpy as np

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
    def __init__(self, env):
        super(Normalize_Observation, self).__init__(env)
        self.observations = []
        self.eps = np.ones(self.env.observation_space.shape) * 1e-6

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.observations = []
        return self.observation(observation)

    def observation(self, observation):
        self.observations.append(observation)
        temp_arr = np.array(self.observations)
        mean = temp_arr.mean(axis=0)
        std = temp_arr.std(axis=0)
        output_observation = (observation - mean) / np.maximum(std, self.eps)
        # print(f"Mean: {output_observation.mean(axis=0)}, Std: {output_observation.std(axis=0)}")
        return output_observation