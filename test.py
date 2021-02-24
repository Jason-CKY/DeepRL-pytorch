import gym
import pybullet_envs
import argparse
import os
import json
import torch
import imageio
import numpy as np

from Wrappers.normalize_observation import Normalize_Observation
from Wrappers.serialize_env import Serialize_Env
from Wrappers.rlbench_wrapper import RLBench_Wrapper
from Wrappers.image_learning import Image_Wrapper

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
    parser.add_argument('--agent', type=str, default='ppo', choices=['ddpg', 'trpo', 'ppo', 'td3', 'option_critic', 'dac_ppo', 'random'], help='specify type of agent')
    parser.add_argument('--render', action='store_true', help='if true, display human renders of the environment')
    parser.add_argument('--gif', action='store_true', help='if true, make gif of the trained agent')
    parser.add_argument('--timesteps', type=int, help='specify number of timesteps to train for')
    parser.add_argument('--seed', type=int, default=0, help='seed number for reproducibility')
    parser.add_argument('--normalize', action='store_true', help='if true, normalize environment observations')
    parser.add_argument('--rlbench', action='store_true', help='if true, use rlbench environment wrappers')
    parser.add_argument('--image', action='store_true', help='if true, use rlbench environment wrappers')
    parser.add_argument('--view', type=str, default='wrist_rgb', 
                        choices=['wrist_rgb', 'front_rgb', 'left_shoulder_rgb', 'right_shoulder_rgb'], 
                        help='choose the type of camera view to generate image (only for RLBench envs)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.rlbench:
        import rlbench.gym
        if args.normalize:
            env_fn = lambda: Normalize_Observation(RLBench_Wrapper(gym.make(args.env), args.view))
        else:
            env_fn = lambda: RLBench_Wrapper(gym.make(args.env, render_mode='rgb_array'), args.view)
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

    save_dir = os.path.join("Model_Weights", args.env, args.agent.lower(), "vae")
    config_path = os.path.join(save_dir, args.agent.lower() + "_config.json")
    logger_kwargs = {
        "output_dir": save_dir
    }
    with open(config_path, 'r') as f:
        model_kwargs = json.load(f)
        # model_kwargs['ac_kwargs']['ngpu'] = 1

    if args.agent.lower() == 'ddpg':
        from Algorithms.ddpg.ddpg import DDPG
        
        model = DDPG(env_fn, save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        model.load_weights(load_buffer=False)

    elif args.agent.lower() == 'td3':
        from Algorithms.td3.td3 import TD3
            
        model = TD3(env_fn, save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        model.load_weights(load_buffer=False)

    elif args.agent.lower() == 'trpo':
        from Algorithms.trpo.trpo import TRPO

        model = TRPO(env_fn, save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        model.load_weights()
    elif args.agent.lower() == 'ppo':
        from Algorithms.ppo.ppo import PPO

        model = PPO(env_fn, save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        model.load_weights()

    elif args.agent.lower() == 'option_critic':
        if not args.rlbench:
            env = env_fn()
            from gym.spaces import Box
            if isinstance(env.action_space, Box):
                from Algorithms.option_critic.oc_continuous import Option_Critic
            else:
                from Algorithms.option_critic.oc_discrete import Option_Critic
            del env
        else:
            from Algorithms.option_critic.oc_continuous import Option_Critic
        model = Option_Critic(env_fn, save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        model.load_weights(fname="latest_1.pth")

    elif args.agent.lower() == 'dac_ppo':
        from Algorithms.dac_ppo.dac_ppo import DAC_PPO
        model = DAC_PPO(env_fn, save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        model.load_weights(fname="best.pth")

    ep_ret, ep_len = model.test(render=args.render, record=args.gif, timesteps=args.timesteps)
    print(f"Episode Return: {ep_ret}\nEpisode Length: {ep_len}")

if __name__=='__main__':
    main()