import gym
import pybullet_envs
import argparse
import os
import json
import torch
import numpy as np

from Wrappers.normalized_action import NormalizedActions

from tqdm import tqdm
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='CartPoleContinuousBulletEnv-v0', help='environment_id')
    parser.add_argument('--agent', type=str, default='ddpg', help='specify type of agent (e.g. DDPG/TRPO/PPO/random)')
    # parser.add_argument('--config_path', type=str, default='Algorithms/ddpg/ddpg_config.json', help='path to config.json')
    parser.add_argument('--timesteps', type=int, required=True, help='specify number of timesteps to train for') 
    parser.add_argument('--seed', type=int, default=0, help='seed number for reproducibility')

    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.agent.lower() == 'ddpg':
        # from Algorithms.ddpg.ddpg import main as main
        from Algorithms.ddpg.ddpg import DDPG
        # main(args.env, args.config_path, args.timesteps, args.seed)  
        config_path = os.path.join("Algorithms", "ddpg", "ddpg_config.json") 
        save_dir = os.path.join("Model_Weights", args.env, "ddpg")
        logger_kwargs = {
            "output_dir": save_dir
        }
        with open(config_path) as f:
            model_kwargs = json.load(f)

        model = DDPG(lambda: gym.make(args.env), save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        model.learn(args.timesteps) 

if __name__=='__main__':
    main()