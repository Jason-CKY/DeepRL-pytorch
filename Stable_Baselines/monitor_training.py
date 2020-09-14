import os

import gym
import pybullet_envs
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from savebest_callback import SaveOnBestTrainingRewardCallback

# Create log dir
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('HumanoidBulletEnv-v0')
# Logs will be saved in log_dir/monitor.csv
env = Monitor(env, log_dir)
# Create action noise because TD3 and DDPG use a deterministic policy
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
# Create RL model
model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=0)
# Train the agent
model.learn(total_timesteps=int(1e6), callback=callback)

