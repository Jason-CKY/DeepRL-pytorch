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
    parser.add_argument('--env', type=str, default='AntBulletEnv-v0', help='environment_id')
    parser.add_argument('--agent', type=str, default='ddpg', help='specify type of agent (e.g. DDPG/TRPO/PPO/random)')
    parser.add_argument('--save_dir', type=str, default='Model_Weights', help='path to store training logs in .json format')
    parser.add_argument('--resume', type=str, help='path to weights to resume training from')
    parser.add_argument('--timesteps', type=int, required=True, help='specify number of timesteps to train for')
    parser.add_argument('--checkpoint_freq', type=int, default=50000, help='number of timesteps before each checkpoint')
  
    return parser.parse_args()

def update_agent_parameters(agent_parameters, env, save_dir):
    agent_parameters['action_space'] = env.action_space
    agent_parameters['observation_space'] = env.observation_space
    agent_parameters['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    agent_parameters['checkpoint_dir'] = save_dir
    return agent_parameters

def main():
    args = parse_arguments()
    # -------- set up environment and monitor logs --------
    save_dir = os.path.join(args.save_dir, args.env, args.agent)
    env = make_vec_env(args.env, n_envs=1, monitor_dir=save_dir)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    # -------- set up agent --------
    if args.agent.lower() == 'ddpg':
        from Agents.DDPG.ddpg_agent import DDPG_Agent
        agent = DDPG_Agent()
        agent_config_path = "config/ddpg_config.json"

    with open(agent_config_path) as f:
        agent_parameters = json.load(f)
        
    agent_parameters = update_agent_parameters(agent_parameters, env, save_dir)
    agent.agent_init(agent_parameters)

    # -------- load checkpoint if any --------
    if args.resume is None:
        starting_timestep = 1
    else:
        agent.load_checkpoint(args.resume)
        starting_timestep = int(args.resume.split(os.path.sep)[-1].split('.')[0].split('_')[-1])


    # -------- training loop --------
    reward_threshold = env.envs[0].spec.reward_threshold
    max_episode_steps = env.envs[0].spec.max_episode_steps
    is_terminal = False
    last_state = env.reset().flatten()                 # flatten() to return the actual state from the vectorized state
    action = agent.agent_start(last_state)
    reward = 0.0
    episode_num = 1
    best_mean_reward = -np.inf
    for timestep in tqdm(range(starting_timestep, args.timesteps+1)):
        if is_terminal or agent.episode_steps > max_episode_steps:
            episode_num += 1
            agent.agent_end(reward)
            # record the episode reward
            episode_reward = agent.sum_rewards
            episode_steps = agent.episode_steps
            # Retrieve training reward
            x, y = ts2xy(load_results(save_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                print("Num timesteps: {}".format(timestep))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

                # New best model
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    agent.save_checkpoint(timestep, best=True)

            if reward_threshold is not None and episode_reward >= reward_threshold:
                print("Environment solved, saving checkpoint")
                agent.save_checkpoint(timestep, solved=True)
                break

            is_terminal = False
            last_state = env.reset().flatten()         # flatten() to return the actual state from the vectorized state
            action = agent.agent_start(last_state)

        else:
            action = np.expand_dims(action, 0)
            state, reward, is_terminal, info = env.step(action)
            action = agent.agent_step(reward.flatten(), state.flatten())       # flatten() to return the actual state from the vectorized state  

        if timestep%args.checkpoint_freq == 0 or timestep == args.timesteps:
            agent.save_checkpoint(timestep)

    env.close()

if __name__=='__main__':
    main()