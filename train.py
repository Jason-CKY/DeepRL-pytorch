import gym
import pybullet_envs
import argparse
import os
import json
import torch
import numpy as np

from tqdm import tqdm
from Wrappers.normalize_observation import Normalize_Observation

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='CartPoleContinuousBulletEnv-v0', help='environment_id')
    parser.add_argument('--agent', type=str, default='ppo', help='specify type of agent (e.g. DDPG/TRPO/PPO/random)')
    parser.add_argument('--timesteps', type=int, required=True, help='specify number of timesteps to train for') 
    parser.add_argument('--seed', type=int, default=0, help='seed number for reproducibility')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of times to train the algo')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if args.agent.lower() == 'ddpg':
        from Algorithms.ddpg.ddpg import DDPG
        config_path = os.path.join("Algorithms", "ddpg", "ddpg_config.json") 
        save_dir = os.path.join("Model_Weights", args.env, "ddpg")
        logger_kwargs = {
            "output_dir": save_dir
        }
        with open(config_path, 'r') as f:
            model_kwargs = json.load(f)

        env_fn = lambda: gym.make(args.env)
        model = DDPG(env_fn, save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        with open(os.path.join(save_dir, "ddpg_config.json"), "w") as f:
            f.write(json.dumps(model_kwargs, indent=4))

    elif args.agent.lower() == 'td3':
        from Algorithms.td3.td3 import TD3
        config_path = os.path.join("Algorithms", "td3", "td3_config.json")
        save_dir = os.path.join("Model_Weights", args.env, "td3")
        logger_kwargs = {
            "output_dir": save_dir
        }
        with open(config_path, 'r') as f:
            model_kwargs = json.load(f)

        env_fn = lambda: gym.make(args.env)
        model = TD3(env_fn, save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        with open(os.path.join(save_dir, "td3_config.json"), "w") as f:
            f.write(json.dumps(model_kwargs, indent=4))        
    
    elif args.agent.lower() == 'trpo':
        from Algorithms.trpo.trpo import TRPO
        config_path = os.path.join("Algorithms", "trpo", "trpo_config.json")
        save_dir = os.path.join("Model_Weights", args.env, "trpo")
        logger_kwargs = {
            "output_dir": save_dir
        }
        with open(config_path, 'r') as f:
            model_kwargs = json.load(f)
        
        env_fn = lambda: Normalize_Observation(gym.make(args.env))
        model = TRPO(env_fn, save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        with open(os.path.join(save_dir, "trpo_config.json"), "w") as f:
            f.write(json.dumps(model_kwargs, indent=4))    

    elif args.agent.lower() == 'ppo':
        from Algorithms.ppo.ppo import PPO
        config_path = os.path.join("Algorithms", "ppo", "ppo_config.json")
        save_dir = os.path.join("Model_Weights", args.env, "ppo")
        logger_kwargs = {
            "output_dir": save_dir
        }
        with open(config_path, 'r') as f:
            model_kwargs = json.load(f)

        env_fn = lambda: Normalize_Observation(gym.make(args.env))
        model = PPO(env_fn, save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        with open(os.path.join(save_dir, "ppo_config.json"), "w") as f:
            f.write(json.dumps(model_kwargs, indent=4))   

    model.learn(args.timesteps, args.num_trials) 

if __name__=='__main__':
    main()