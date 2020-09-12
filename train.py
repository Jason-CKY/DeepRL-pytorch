import gym
import pybullet_envs
import argparse
import os
import json
import torch
import numpy as np

from Logger.logger import Logger
from tqdm import tqdm
from Wrappers.normalized_action import NormalizedActions

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HumanoidBulletEnv-v0', help='environment_id')
    parser.add_argument('--agent', type=str, default='ddpg', help='specify type of agent (e.g. DDPG/TRPO/PPO/random)')
    parser.add_argument('--log_dir', type=str, default='Logger/logs', help='path to store training logs in .json format')
    parser.add_argument('--resume', type=str, help='path to weights to resume training from')
    parser.add_argument('--timesteps', type=int, required=True, help='specify number of timesteps to train for')
    parser.add_argument('--checkpoint_freq', type=int, default=50000, help='number of timesteps before each checkpoint')
    parser.add_argument('--print_freq', type=int, default=5, help='number of episodes before printing progress')
  
    return parser.parse_args()

def update_agent_parameters(agent_parameters, env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent_parameters['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    agent_parameters['network_config']['state_dim'] = state_dim
    agent_parameters['network_config']['action_dim'] = action_dim

    checkpoint_dir = agent_parameters['checkpoint_dir']
    agent_parameters['checkpoint_dir'] = os.path.join(checkpoint_dir, env.spec.id)
    return agent_parameters

def main():
    args = parse_arguments()
    # -------- set up environment --------
    env = NormalizedActions(gym.make(args.env))

    # -------- set up agent --------
    if args.agent.lower() == 'ddpg':
        from Agents.DDPG.ddpg_agent import DDPG_Agent
        agent = DDPG_Agent()
        agent_config_path = "config/ddpg_config.json"

    with open(agent_config_path) as f:
        agent_parameters = json.load(f)
        
    agent_parameters = update_agent_parameters(agent_parameters, env)
    agent.agent_init(agent_parameters)

    # Set up logs for the training session
    logger = Logger(agent_parameters, env)

    # -------- load checkpoint if any --------
    if args.resume is None:
        starting_timestep = 1
    else:
        agent.load_checkpoint(args.resume)
        starting_timestep = int(args.resume.split(os.path.sep)[-1].split('.')[0].split('_')[-1])


    # -------- training loop --------
    reward_threshold = env.spec.reward_threshold
    max_episode_steps = env.spec.max_episode_steps
    is_terminal = False
    last_state = env.reset()
    action = agent.agent_start(last_state)
    reward = 0.0
    episode_num = 1
    max_episode_reward = -np.inf
    for timestep in tqdm(range(starting_timestep, args.timesteps+1)):
        if is_terminal or agent.episode_steps > max_episode_steps:
            episode_num += 1
            agent.agent_end(reward)
            # record the episode reward
            episode_reward = agent.sum_rewards
            episode_steps = agent.episode_steps
            logger.log_episode(episode_reward, episode_steps)
            if args.print_freq%episode_num:
                print('Timestep {}/{} | Episode_Reward {}'.format(timestep, args.timesteps, episode_reward))
            if reward_threshold is not None and episode_reward >= reward_threshold:
                print("Environment solved, saving checkpoint")
                agent.save_checkpoint(timestep, solved=True)
                break
            if episode_reward >= max_episode_reward:
                agent.save_checkpoint(timestep, best=True)

            is_terminal = False
            last_state = env.reset()
            action = agent.agent_start(last_state)

        else:
            state, reward, is_terminal, info = env.step(action)
            agent.agent_step(reward, state)         

        if timestep%args.checkpoint_freq == 0 or timestep == args.timesteps:
            agent.save_checkpoint(timestep)


    env.close()

if __name__=='__main__':
    main()