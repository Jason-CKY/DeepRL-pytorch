import gym
import pybullet_envs
import argparse
import os
import json
import torch
import numpy as np

from tqdm import tqdm
from Wrappers.normalize_observation import Normalize_Observation
from Wrappers.serialize_env import Serialize_Env
from Wrappers.rlbench_wrapper import RLBench_Wrapper

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='CartPoleContinuousBulletEnv-v0', help='environment_id')
    parser.add_argument('--agent', type=str, default='ppo', choices=['ddpg', 'trpo', 'ppo', 'td3', 'random'], help='specify type of agent')
    parser.add_argument('--arch', type=str, default='mlp', choices=['mlp', 'cnn'], help='specify architecture of neural net')
    parser.add_argument('--timesteps', type=int, required=True, help='specify number of timesteps to train for') 
    parser.add_argument('--seed', type=int, default=0, help='seed number for reproducibility')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of times to train the algo')
    parser.add_argument('--normalize', action='store_true', help='if true, normalize environment observations')
    parser.add_argument('--rlbench', action='store_true', help='if true, use rlbench environment wrappers')

    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.rlbench:
        import rlbench.gym
        if args.normalize:
            env_fn = lambda: Normalize_Observation(RLBench_Wrapper(gym.make(args.env), 'wrist_rgb'))
        else:
            env_fn = lambda: RLBench_Wrapper(gym.make(args.env), 'wrist_rgb')
    elif args.normalize:
        env_fn = lambda: Normalize_Observation(gym.make(args.env))
    else:
        env_fn = lambda: Serialize_Env(gym.make(args.env))

    config_path = os.path.join("Algorithms", args.agent.lower(), args.agent.lower() + "_config_" + args.arch + ".json")
    save_dir = os.path.join("Model_Weights", args.env, args.agent.lower())
    logger_kwargs = {
        "output_dir": save_dir
    }
    with open(config_path, 'r') as f:
        model_kwargs = json.load(f)

    if args.agent.lower() == 'ddpg':
        from Algorithms.ddpg.ddpg import DDPG
        if args.arch == 'mlp':
            from Algorithms.ddpg.core import MLPActorCritic
            ac = MLPActorCritic
        elif args.arch == 'cnn':
            from Algorithms.ddpg.core import CNNActorCritic
            ac = CNNActorCritic

        model = DDPG(env_fn, save_dir, actor_critic=ac, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        with open(os.path.join(save_dir, "ddpg_config.json"), "w") as f:
            f.write(json.dumps(model_kwargs, indent=4))

    elif args.agent.lower() == 'td3':
        from Algorithms.td3.td3 import TD3

        model = TD3(env_fn, save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        with open(os.path.join(save_dir, "td3_config.json"), "w") as f:
            f.write(json.dumps(model_kwargs, indent=4))        
    
    elif args.agent.lower() == 'trpo':
        from Algorithms.trpo.trpo import TRPO
        
        model = TRPO(env_fn, save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        with open(os.path.join(save_dir, "trpo_config.json"), "w") as f:
            f.write(json.dumps(model_kwargs, indent=4))    

    elif args.agent.lower() == 'ppo':
        from Algorithms.ppo.ppo import PPO

        model = PPO(env_fn, save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        with open(os.path.join(save_dir, "ppo_config.json"), "w") as f:
            f.write(json.dumps(model_kwargs, indent=4))   

    model.learn(args.timesteps, args.num_trials) 

if __name__=='__main__':
    main()