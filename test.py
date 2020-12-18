import gym
import pybullet_envs
import argparse
import os
import json
import torch
import imageio
import numpy as np

from Wrappers.normalize_observation import Normalize_Observation
# from stable_baselines3.common.cmd_util import make_vec_env
# from stable_baselines3.common.vec_env import VecNormalize

def random_test(env_fn, render=True, record_dir=None, timesteps=None):
    '''
    Perform a random walkthrough of the environment.
    Args:
        env_fn (function): function call to create the environment
        render (bool): If true, render the image out for user to see in real time
        record_dir (str): Path to save the recorded gif. Default None will not save the recording of the episode
        timesteps (int): number of timesteps to run the environment for. Default None will run to completion
    '''
    env = env_fn()
    if render:
        env.render('human')
    state, done, ep_ret, ep_len = env.reset(), False, 0, 0
    img = []
    if record_dir is not None:
        img.append(env.render('rgb_array'))

    if timesteps is not None:
        for i in range(timesteps):
            # Take deterministic action with 0 noise added
            state, reward, done, _ = env.step(env.action_space.sample())
            if record_dir is not None:
                img.append(env.render('rgb_array'))
            else:
                env.render()
            ep_ret += reward
            ep_len += 1                
    else:
        while not done:
            # Take deterministic action with 0 noise added
            state, reward, done, _ = env.step(env.action_space.sample())
            if record_dir is not None:
                img.append(env.render('rgb_array'))
            else:
                env.render()
            ep_ret += reward
            ep_len += 1

    if record_dir is not None:
        imageio.mimsave(f'{os.path.join(record_dir, "random_recording.gif")}', [np.array(img) for i, img in enumerate(img) if i%2 == 0], fps=29)

    return ep_ret, ep_len  

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='AntBulletEnv-v0', help='environment_id')
    parser.add_argument('--agent', type=str, default='ppo', choices=['ddpg', 'trpo', 'ppo', 'td3', 'random'], help='specify type of agent')
    parser.add_argument('--arch', type=str, default='mlp', choices=['mlp', 'cnn'], help='specify architecture of neural net')
    parser.add_argument('--render', action='store_true', help='if true, display human renders of the environment')
    parser.add_argument('--gif', action='store_true', help='if true, make gif of the trained agent')
    parser.add_argument('--timesteps', type=int, help='specify number of timesteps to train for')
    parser.add_argument('--seed', type=int, default=0, help='seed number for reproducibility')
    parser.add_argument('--normalize', action='store_true', help='if true, normalize environment observations')
    parser.add_argument('--rlbench', action='store_true', help='if true, use rlbench environment wrappers')
    parser.add_argument('--image', action='store_true', help='if true, use rlbench environment wrappers')

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
    elif args.image:
        env_fn = lambda: Image_Wrapper(gym.make(args.env))
    else:
        env_fn = lambda: Serialize_Env(gym.make(args.env))
    
    if args.agent.lower() == 'random':
        save_dir = os.path.join("Model_Weights", args.env) if args.gif else None
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        random_test(env_fn, render=args.render, record_dir=save_dir, timesteps=args.timesteps)
        return

    save_dir = os.path.join("Model_Weights", args.env, args.agent.lower())
    config_path = os.path.join(save_dir, args.agent.lower() + "_config.json")
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
        model.load_weights(load_buffer=False)

    elif args.agent.lower() == 'td3':
        from Algorithms.td3.td3 import TD3
        if args.arch == 'mlp':
            from Algorithms.td3.core import MLPActorCritic
            ac = MLPActorCritic
        elif args.arch == 'cnn':
            from Algorithms.td3.core import CNNActorCritic
            ac = CNNActorCritic
            
        model = TD3(env_fn, save_dir, actor_critic=ac, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        model.load_weights(load_buffer=False)

    elif args.agent.lower() == 'trpo':
        from Algorithms.trpo.trpo import TRPO
        if args.arch == 'mlp':
            from Algorithms.trpo.core import MLPActorCritic
            ac = MLPActorCritic
        elif args.arch == 'cnn':
            from Algorithms.trpo.core import CNNActorCritic
            ac = CNNActorCritic

        model = TRPO(env_fn, save_dir, actor_critic=ac, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        model.load_weights()
    elif args.agent.lower() == 'ppo':
        from Algorithms.ppo.ppo import PPO
        if args.arch == 'mlp':
            from Algorithms.ppo.core import MLPActorCritic
            ac = MLPActorCritic
        elif args.arch == 'cnn':
            from Algorithms.ppo.core import CNNActorCritic
            ac = CNNActorCritic

        model = PPO(env_fn, save_dir, actor_critic=ac, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        model.load_weights()

    ep_ret, ep_len = model.test(render=args.render, record=args.gif, timesteps=args.timesteps)
    print(f"Episode Return: {ep_ret}\nEpisode Length: {ep_len}")

if __name__=='__main__':
    main()