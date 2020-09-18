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
    parser.add_argument('--save_dir', type=str, default='Model_Weights', help='path to store training logs in .json format')
    parser.add_argument('--render', action='store_true', help='if true, display human renders of the environment')
    parser.add_argument('--gif', action='store_true', help='if true, make gif of the trained agent')
    parser.add_argument('--load_latest', action='store_true', help='if true, load the latest saved model from the checkpoint directory')
    parser.add_argument('--load', type=str, help='specify load path')
    parser.add_argument('--timesteps', type=int, help='specify number of timesteps to train for')

    return parser.parse_args()

def update_agent_parameters(agent_parameters, env, save_dir):
    agent_parameters['action_space'] = env.action_space
    agent_parameters['observation_space'] = env.observation_space
    agent_parameters['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    agent_parameters['checkpoint_dir'] = save_dir
    return agent_parameters

def main():
    args = parse_arguments()
    save_dir = os.path.join(args.save_dir, args.env, args.agent)
    env = make_vec_env(args.env, n_envs=1)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    #  do not update them at test time
    env.training = False
    # reward normalization is not needed at test time
    env.norm_reward = False

    if args.render:
        env.render('human')

    if args.agent == 'random':
        from Agents.random_agent import Random_Agent
        agent = Random_Agent()
        agent_config_path = "Config/random_config.json"
    elif args.agent == 'ddpg':
        from Agents.DDPG.ddpg_agent import DDPG_Agent
        agent = DDPG_Agent()
        agent_config_path = "Config/ddpg_config.json"

    with open(agent_config_path) as f:
        agent_parameters = json.load(f)
    agent_parameters = update_agent_parameters(agent_parameters, env, save_dir)
    agent.agent_init(agent_parameters)

    if args.load_latest:
        agent.load_checkpoint()
    elif args.load is not None:
        agent.load_checkpoint(args.load)

    last_state = env.reset()
    action = agent.policy(last_state)
    done = False
 
    if args.timesteps is None:
        while not done: 
            state, reward, done, info = env.step(action)
            action = agent.policy(state)
            last_state = state
            env.render()
    else:
        for i in range(args.timesteps): 
            state, reward, done, info = env.step(action)
            action = agent.policy(state)
            last_state = state
            env.render()


    env.close()

if __name__=='__main__':
    main()