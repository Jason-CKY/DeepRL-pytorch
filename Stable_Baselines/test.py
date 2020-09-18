import gym
import pybullet_envs
import argparse
import os
import torch
import numpy as np
import imageio

from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import torch.nn as nn

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='AntBulletEnv-v0', help='environment_id')
    parser.add_argument('--agent', type=str, default='ddpg', help='specify type of agent (e.g. DDPG/TRPO/PPO/random)')
    parser.add_argument('--log_dir', type=str, default='logs', help='path to store training logs in .json format')
    parser.add_argument('--gif', action='store_true', help='if true, make gif of the trained agent')
    parser.add_argument('--render', action='store_true', help='if true, display human renders of the environment')
    parser.add_argument('--timesteps', type=int, help='specify number of timesteps to train for')

    return parser.parse_args()

def main():
    args = parse_arguments()
    load_path = os.path.join("logs", args.env, args.agent, "best_model.zip")
    stats_path = os.path.join(args.log_dir, args.env, args.agent, "vec_normalize.pkl")

    if args.agent == 'ddpg':
        from stable_baselines3 import DDPG
        model = DDPG.load(load_path)
    elif args.agent == 'td3':
        from stable_baselines3 import TD3
        model = TD3.load(load_path)
    elif args.agent == 'ppo':
        from stable_baselines3 import PPO
        model = PPO.load(load_path)

    env = make_vec_env(args.env, n_envs=1)
    env = VecNormalize.load(stats_path, env)
    #  do not update them at test time
    env.training = False
    # reward normalization is not needed at test time
    env.norm_reward = False
    
    # env = gym.make(args.env)
    img = []
    if args.render:
        env.render('human')
    done = False
    obs = env.reset()
    action = model.predict(obs)
    if args.gif:
        img.append(env.render('rgb_array'))

    if args.timesteps is None:
        while not done: 
            action, _= model.predict(obs)
            obs, reward, done, info = env.step(action)
            if args.gif:
                img.append(env.render('rgb_array'))
            else:
                env.render()
    else:
        for i in range(args.timesteps): 
            action, _= model.predict(obs)
            obs, reward, done, info = env.step(action)
            if args.gif:
                img.append(env.render('rgb_array'))
            else:
                env.render()

    if args.gif:
        imageio.mimsave(f'{os.path.join("logs", args.env, args.agent, "recording.gif")}', [np.array(img) for i, img in enumerate(img) if i%2 == 0], fps=29)
        
if __name__ == '__main__':
    main()