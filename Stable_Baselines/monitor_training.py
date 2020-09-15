import os

import gym
import pybullet_envs
import numpy as np
import argparse
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from savebest_callback import SaveOnBestTrainingRewardCallback


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HumanoidBulletEnv-v0', help='environment_id')
    parser.add_argument('--agent', type=str, default='ddpg', help='specify type of agent (e.g. DDPG/TRPO/PPO/random)')
    parser.add_argument('--log_dir', type=str, default='logs', help='path to store training logs in .json format')
    parser.add_argument('--timesteps', type=int, required=True, help='specify number of timesteps to train for')
  
    return parser.parse_args()

def main():
    args = parse_arguments()
    # Create log dir
    log_dir = os.path.join(args.log_dir, args.env, args.agent)
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make(args.env)
    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)
    # Create action noise because TD3 and DDPG use a deterministic policy
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    # Create RL model
    if args.agent == 'ddpg':
        model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=0)

    # Train the agent
    model.learn(total_timesteps=args.timesteps, callback=callback)

if __name__ == '__main__':
    main()