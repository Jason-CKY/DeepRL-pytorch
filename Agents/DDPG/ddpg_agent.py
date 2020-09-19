from ..base_agent import BaseAgent
from ..replay_buffer import ReplayBuffer
from .ddpg_nets import Actor, Critic
from .ounoise import OUNoise
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import os

from copy import deepcopy
from stable_baselines3.common.noise import NormalActionNoise
class DDPG_Agent(BaseAgent):
    def __init__(self):
        pass
    
    def agent_init(self, agent_config):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the agent.

        Assume agent_config dict contains:
        {
            network_config: dictionary,
            optimizer_config: dictionary,
            replay_buffer_size: integer,
            minibatch_sz: integer, 
            num_replay_updates_per_step: float
            discount_factor: float,
            checkpoint_dir: str
        }
        """
        self.name = agent_config['name']
        self.device = agent_config['device']
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'],
                                        agent_config['minibatch_size'],
                                        agent_config.get('seed'))
        self.action_space = agent_config['action_space']
        self.obs_space = agent_config['observation_space']
        agent_config['network_config']['state_dim'] = self.obs_space.shape[-1]
        agent_config['network_config']['action_dim'] = self.action_space.shape[-1]
        # define network
        self.actor = Actor(agent_config['network_config']).to(self.device)
        self.actor_target = deepcopy(self.actor).to(self.device)
        # self.actor_target = Actor(agent_config['network_config']).to(self.device)
        # self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(agent_config['network_config']).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)
        # self.critic_target = Critic(agent_config['network_config']).to(self.device)
        # self.critic_target.load_state_dict(self.critic.state_dict())

        optim_config = agent_config['optimizer_config']
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=optim_config['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=optim_config['critic_lr'])
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.update_after = agent_config['update_after']
        self.discount = agent_config['gamma']
        self.tau = agent_config['tau']

        self.noise_type = agent_config['noise']
        n_actions = self.action_space.shape[-1]
        if self.noise_type == 'OUNoise':
            self.noise = OUNoise(n_actions)
        else:
            self.noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        self.rand_generator = np.random.RandomState(agent_config.get('seed'))

        self.last_state = None
        self.last_action = None

        self.sum_rewards = 0
        self.episode_steps = 0

        checkpoint_dir = agent_config.get('checkpoint_dir')
        if checkpoint_dir is None:
            self.checkpoint_dir = 'model_weights'
        else:
            self.checkpoint_dir = checkpoint_dir
        
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def optimize_network(self, experiences):
        """
        Args:
            experiences (Numpy array): The batch of experiences including the states, actions, 
                                    rewards, terminals, and next_states.
        """
        self.set_train()
        # Get states, action, rewards, terminals, and next_states from experiences
        states, actions, rewards, terminals, next_states = experiences
        states = torch.tensor(states).to(self.device).float()
        next_states = torch.tensor(next_states).to(self.device).float()
        actions = torch.tensor(actions).to(self.device).float()
        rewards = torch.tensor(rewards).to(self.device).float()
        terminals = torch.tensor(terminals).to(self.device).float()
        
        # ------------------- optimize critic ----------------------------
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_Q = self.critic_target(next_states, next_actions) * (1-terminals)
            Qprime = rewards + (self.discount * next_Q)

        Qvals = self.critic(states, actions)
        assert Qvals.shape == Qprime.shape
        critic_loss = F.mse_loss(Qvals, Qprime)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------- optimize actor ----------------------------
        policy_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def policy(self, state, add_noise=True):
        """
        Args:
            state (Numpy array)/(torch tensor)/(list): the state
        Returns:
            the action
        if state is in the shape [n, state_dim], output will be [n, action_dim]
        if state is in the shape [state_dim], output will be [action_dim]
        """
        self.set_eval()
        state = torch.tensor(state).to(self.device).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
            with torch.no_grad():
                action = self.actor(state).cpu().detach().numpy()[0]
        else:
            with torch.no_grad():
                action = self.actor(state).cpu().detach().numpy()

        if add_noise:                                       # add noise
            if self.noise_type == 'OUNoise':
                action = self.noise.get_action(action)  
            else:
                action += self.noise()

        action = np.clip(action*self.action_space.high, self.action_space.low, self.action_space.high)                             # clip to tanh range [-1, 1]
        return action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """
        self.noise.reset()
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array(state)
        self.last_action = self.policy(self.last_state)
        return self.last_action

    def agent_step(self, reward, state, exploration=False):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        self.sum_rewards += reward
        self.episode_steps += 1

        state = np.array(state)
        if exploration:
            action = self.action_space.sample()
        else:
            action = self.policy(state)
        
        # Append new experience to replay buffer
        self.replay_buffer.append(self.last_state, self.last_action, reward, 0, state)
        
        # Perform replay steps:
        if self.replay_buffer.size() > self.update_after:
            for _ in range(self.num_replay):
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()     
                self.optimize_network(experiences)
                
        # Update the last state and last action.
        self.last_state = state
        self.last_action = action
        
        return action
        
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1
        
        # Set terminal state to an array of zeros
        state = np.zeros_like(self.last_state)

        # Append new experience to replay buffer
        self.replay_buffer.append(self.last_state, self.last_action, reward, 1, state)
        
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            for _ in range(self.num_replay):
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()
                self.optimize_network(experiences)
                

    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        pass

    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")

    def set_train(self):
        '''
        Set actor and critic networks into train mode
        '''
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def set_eval(self):
        '''
        Set actor and critic networks into eval mode
        '''
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
    
    def save_checkpoint(self, timesteps, solved=False, best=False):
        """Saving networks and optimizer paramters to a file in 'checkpoint_dir'
        Args:
            timesteps: number of timesteps the model has been trained on
        """
        if solved:
            checkpoint_name = os.path.join(self.checkpoint_dir, "solved.pth")
        elif best:
            checkpoint_name = os.path.join(self.checkpoint_dir, "best.pth")
        else:
            checkpoint_name = os.path.join(self.checkpoint_dir, f"timestep_{timesteps}.pth")
        
        print('saving checkpoint...')
        checkpoint = {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_name)
        self.replay_buffer.save(os.path.join(self.checkpoint_dir, "replay_buffer.pickle"))
        print(f"checkpoint saved at {checkpoint_name}")
    
    def get_latest_path(self):
        """
        get the latest created file in the checkpoint directory
        Returns:
            the latest saved model weights
        """
        files = [fname for fname in os.listdir(self.checkpoint_dir) if fname.endswith(".pth")]
        filepaths = [os.path.join(self.checkpoint_dir, filepath) for filepath in files]
        latest_file = max(filepaths, key=os.path.getctime)
        return latest_file
        
    def load_checkpoint(self, fname=None, load_buffer=False):
        """
        load networks and optimizer paramters from checkpoint_path
        if checkpoint_path is None, use the latest created path from checkpoint_dir
        Args:
            checkpoint_path: path to checkpoint
        """
        if fname is None:
            fname = self.get_latest_path()

        checkpoint_path = os.path.join(self.checkpoint_dir, fname)
        if os.path.isfile(checkpoint_path):
            if load_buffer:
                self.replay_buffer.load(os.path.join(self.checkpoint_dir, "replay_buffer.pickle"))
            key = 'cuda' if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(checkpoint_path, map_location=key)
            self.actor.load_state_dict(checkpoint['actor'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])

            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

            print('checkpoint loaded at {}'.format(checkpoint_path))
        else:
            raise OSError("Checkpoint file not found.")    

