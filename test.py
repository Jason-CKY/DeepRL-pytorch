import gym
import pybullet_envs
import argparse
import os
import json
import torch

from Wrappers.normalized_action import NormalizedActions
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='AntBulletEnv-v0', help='environment_id')
    parser.add_argument('--agent', type=str, default='ddpg', help='specify type of agent (e.g. DDPG/TRPO/PPO/random)')
    parser.add_argument('--render', action='store_true', help='if true, display human renders of the environment')
    parser.add_argument('--gif', action='store_true', help='if true, make gif of the trained agent')
    parser.add_argument('--timesteps', type=int, help='specify number of timesteps to train for')
    parser.add_argument('--seed', type=int, default=0, help='seed number for reproducibility')

    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.agent.lower() == 'ddpg':
        # from Algorithms.ddpg.ddpg import main as main
        from Algorithms.ddpg.ddpg import DDPG
        # main(args.env, args.config_path, args.timesteps, args.seed)  
        save_dir = os.path.join("Model_Weights", args.env, "ddpg")
        config_path = os.path.join(save_dir, "ddpg_config.json") 
        logger_kwargs = {
            "output_dir": save_dir
        }
        with open(config_path) as f:
            model_kwargs = json.load(f)

        model = DDPG(lambda: gym.make(args.env), save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        model.load_weights(load_buffer=False)
    model.test(render=args.render, record=args.gif)

if __name__=='__main__':
    main()