import os

import gym
import pybullet_envs
import numpy as np
import argparse

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

from savebest_callback import SaveOnBestTrainingRewardCallback


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='AntBulletEnv-v0', help='environment_id')
    parser.add_argument('--agent', type=str, default='ddpg', help='specify type of agent (e.g. DDPG/TRPO/PPO/random)')
    parser.add_argument('--log_dir', type=str, default='logs', help='path to store training logs and weights')
    parser.add_argument('--timesteps', type=int, required=True, help='specify number of timesteps to train for')
  
    return parser.parse_args()

def main():
    args = parse_arguments()
    # Create log dir
    log_dir = os.path.join(args.log_dir, args.env, args.agent)
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    # env = make_vec_env(args.env, n_envs=1, monitor_dir=log_dir)
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    env = gym.make(args.env)
    env = Monitor(env, log_dir)
    # Create action noise because TD3 and DDPG use a deterministic policy
    n_actions = env.action_space.shape[-1]
    
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    # Create RL model
    if args.agent == 'ddpg':
        from stable_baselines3 import DDPG
        policy_kwargs = dict(net_arch=[64, 64])
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = DDPG('MlpPolicy', env, action_noise=action_noise, policy_kwargs=policy_kwargs, verbose=0)
    elif args.agent == 'td3':   
        from stable_baselines3 import TD3
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=0)
    elif args.agent == 'ppo':
        from stable_baselines3 import PPO
        model = PPO('MlpPolicy', env, verbose=0)

    # Train the agent
    model.learn(total_timesteps=args.timesteps, callback=callback)

if __name__ == '__main__':
    main()