import json
import gym
import os
import time

class Logger():
    def __init__(self, agent_parameters:dict, env:gym.Env, log_path='Logger/logs'):
        self.logger = {
            'reward_timestep_time': []
        }
        self.logger['agent_parameters'] = agent_parameters
        self.logger['env_parameters'] = {
            'env_id': env.spec.id,
            'reward_threshold': env.spec.reward_threshold,
            'max_episode_steps': env.spec.max_episode_steps
        }

        self.t_start = time.time()
        self.logfile = os.path.join(log_path, '_'.join([agent_parameters['name'], env.spec.id, str(int(time.time()))])+'.json')
        os.makedirs(log_path, exist_ok=True)
        self.write_to_file()

    def log_episode(self, episode_reward, episode_steps):
        self.logger['reward_timestep_time'].append([episode_reward, episode_steps, time.time()-self.t_start])
        self.t_start = time.time()
        self.write_to_file()

    def write_to_file(self):
        with open(self.logfile, 'w') as f:
            f.write(json.dumps(self.logger, indent=4, sort_keys=True))

    def load_logger(self):
        with open(self.logfile, 'r') as f:
            self.logger = json.load(f)