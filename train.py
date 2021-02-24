import gym
import pybullet_envs
import argparse
import os
import json

from Wrappers.normalize_observation import Normalize_Observation
from Wrappers.serialize_env import Serialize_Env
from Wrappers.rlbench_wrapper import RLBench_Wrapper
from Wrappers.image_learning import Image_Wrapper

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='CartPoleContinuousBulletEnv-v0', help='environment_id')
    parser.add_argument('--agent', type=str, default='ppo', choices=['ddpg', 'trpo', 'ppo', 'td3', 'option_critic', 'dac_ppo', 'random'], help='specify type of agent')
    parser.add_argument('--timesteps', type=int, required=True, help='specify number of timesteps to train for') 
    parser.add_argument('--seed', type=int, default=0, help='seed number for reproducibility')
    parser.add_argument('--ngpu', type=int, default=1, help='Number of gpus to use for training (default 1)')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of times to train the algo')
    parser.add_argument('--normalize', action='store_true', help='if true, normalize environment observations')
    parser.add_argument('--rlbench', action='store_true', help='if true, use rlbench environment wrappers')
    parser.add_argument('--image', action='store_true', help='if true, use image environment wrappers')
    parser.add_argument('--view', type=str, default='front_rgb', 
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
            env_fn = lambda: RLBench_Wrapper(gym.make(args.env), args.view)
    elif args.normalize:
        env_fn = lambda: Normalize_Observation(gym.make(args.env))
    elif args.image:
        env_fn = lambda: Image_Wrapper(gym.make(args.env))
    else:
        env_fn = lambda: Serialize_Env(gym.make(args.env))
        
    config_path = os.path.join("Algorithms", args.agent.lower(), args.agent.lower() + "_config.json")
    save_dir = os.path.join("Model_Weights", args.env, args.agent.lower())
    logger_kwargs = {
        "output_dir": save_dir
    }
    with open(config_path, 'r') as f:
        model_kwargs = json.load(f)
        model_kwargs['ngpu'] = args.ngpu
        
    if args.agent.lower() == 'ddpg':
        from Algorithms.ddpg.ddpg import DDPG

        model = DDPG(env_fn, save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
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
        save_dir = os.path.join(save_dir, model_kwargs['oc_kwargs']['model_type'])
        os.makedirs(save_dir, exist_ok=True)
        logger_kwargs = {
            "output_dir": save_dir
        }
        model_kwargs['tensorboard_logdir'] = os.path.join("tf_logs", args.env, args.agent, model_kwargs['oc_kwargs']['model_type'])
        model = Option_Critic(env_fn, save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        with open(os.path.join(save_dir, "option_critic_config.json"), "w") as f:
            f.write(json.dumps(model_kwargs, indent=4))   

    elif args.agent.lower() == 'dac_ppo':
        from Algorithms.dac_ppo.dac_ppo import DAC_PPO
        model_kwargs['tensorboard_logdir'] = os.path.join("tf_logs", args.env, args.agent)
        model = DAC_PPO(env_fn, save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
        with open(os.path.join(save_dir, "dac_ppo_config.json"), "w") as f:
            f.write(json.dumps(model_kwargs, indent=4))   

    # print(model.network)
    model.learn(args.timesteps, args.num_trials) 

if __name__=='__main__':
    main()