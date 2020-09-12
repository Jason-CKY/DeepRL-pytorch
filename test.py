import gym
import pybullet_envs
import argparse
import os
import json

from Wrappers.normalized_action import NormalizedActions

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HumanoidBulletEnv-v0', help='environment_id')
    parser.add_argument('--agent', type=str, default='random', help='specify type of agent (e.g. DDPG/TRPO/PPO/random)')
    parser.add_argument('--render', action='store_true', help='if true, display human renders of the environment')
    parser.add_argument('--load_latest', action='store_true', help='if true, load the latest saved model from the checkpoint directory')
    parser.add_argument('--load', type=str, help='specify load path')
    parser.add_argument('--timesteps', type=int, help='specify number of timesteps to train for')

    return parser.parse_args()

def update_agent_parameters(agent_parameters, env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if agent_parameters['name'] == 'random_policy':
        agent_parameters['action_space'] = env.action_space
    agent_parameters['network_config']['state_dim'] = state_dim
    agent_parameters['network_config']['action_dim'] = action_dim

    checkpoint_dir = agent_parameters['checkpoint_dir']
    agent_parameters['checkpoint_dir'] = os.path.join(checkpoint_dir, env.unwrapped.spec.id)
    return agent_parameters

def main():
    args = parse_arguments()
    env = NormalizedActions(gym.make(args.env))
    if args.render:
        env.render('human')

    if args.agent == 'random':
        from Agents.random_agent import Random_Agent
        agent = Random_Agent()
        agent_config_path = "config/random_config.json"
    elif args.agent == 'ddpg':
        from Agents.DDPG.ddpg_agent import DDPG_Agent
        agent = DDPG_Agent()
        agent_config_path = "config/ddpg_config.json"

    with open(agent_config_path) as f:
        agent_parameters = json.load(f)
    agent_parameters = update_agent_parameters(agent_parameters, env)
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