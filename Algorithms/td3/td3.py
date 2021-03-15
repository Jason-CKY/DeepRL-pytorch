import gym
import pybullet_envs
import torch
import numpy as np
import time
import argparse
import os
import imageio

from Algorithms.td3.core import MLPActorCritic, CNNActorCritic
from Algorithms.utils import get_actor_critic_module, sanitise_state_dict
from Algorithms.td3.replay_buffer import ReplayBuffer
from Logger.logger import Logger
from copy import deepcopy
from torch.optim import Adam
from tqdm import tqdm
from itertools import chain
from torch.utils.tensorboard import SummaryWriter

class TD3:
    def __init__(self, env_fn, save_dir, ac_kwargs=dict(), seed=0, tensorboard_logdir=None,
         replay_size=int(1e6), gamma=0.99, 
         tau=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
         update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, 
         max_ep_len=1000, logger_kwargs=dict(), save_freq=1, policy_delay=2, ngpu=1):    
        '''
        Twin Delayed Deep Deterministic Policy Gradients (TD3):
        An Extension of DDPG but with 3 tricks added:
            (1) Clipped Double Q-Learning: TD3 learns two Q-functions instead of one (hence “twin”),
                and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions
            (2) “Delayed” Policy Updates. TD3 updates the policy (and target networks) less frequently 
                than the Q-function. The paper recommends one policy update for every two Q-function updates.
            (3) Target Policy Smoothing. TD3 adds noise to the target action, to make it harder for the policy
                to exploit Q-function errors by smoothing out Q along changes in action.
        Args:
            env_fn: function to create the gym environment
            save_dir: path to save directory
            actor_critic: Class for the actor-critic pytorch module
            ac_kwargs (dict): any keyword argument for the actor_critic
                        (1) hidden_sizes=(256, 256)
                        (2) activation=nn.ReLU
                        (3) device='cpu'
            seed (int): seed for random generators
            replay_size (int): Maximum length of replay buffer.
            gamma (float): Discount factor. (Always between 0 and 1.)
            tau (float): Interpolation factor in polyak averaging for target 
                networks.
            pi_lr (float): Learning rate for policy.
            q_lr (float): Learning rate for Q-networks.
            batch_size (int): Minibatch size for SGD.
            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.
            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.
            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long 
                you wait between updates, the ratio of env steps to gradient steps 
                is locked to 1.
            act_noise (float): Stddev for Gaussian exploration noise added to 
                policy at training time. (At test time, no noise is added.)
            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.
            max_ep_len (int): Maximum length of trajectory / episode / rollout.
            logger_kwargs (dict): Keyword args for Logger. 
                        (1) output_dir = None
                        (2) output_fname = 'progress.pickle'
            save_freq (int): How often (in terms of gap between episodes) to save
                    the current policy and value function.
            policy_delay (int): Policy will only be updated once every 
                                policy_delay times for each update of the Q-networks.
        '''
        # logger stuff
        self.logger = Logger(**logger_kwargs)

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = env_fn()

        # Action Limit for clamping
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module
        self.ngpu = ngpu
        self.actor_critic = get_actor_critic_module(ac_kwargs, 'td3')
        self.ac_kwargs = ac_kwargs
        self.ac = self.actor_critic(self.env.observation_space, self.env.action_space, device=self.device, ngpu=self.ngpu, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        
        # Experience buffer
        self.replay_size = replay_size
        self.replay_buffer = ReplayBuffer(int(replay_size))

        # Set up optimizers for actor and critic
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(chain(self.ac.q1.parameters(), self.ac.q2.parameters()), lr=q_lr)

        self.gamma = gamma
        self.tau = tau
        self.act_noise = act_noise
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = self.env.spec.max_episode_steps if self.env.spec.max_episode_steps is not None else max_ep_len
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.policy_delay = policy_delay

        self.best_mean_reward = -np.inf
        self.save_dir = save_dir
        self.tensorboard_logdir = tensorboard_logdir

    def reinit_network(self):
        '''
        Re-initialize network weights and optimizers for a fresh agent to train
        '''
        # Create actor-critic module
        self.best_mean_reward = -np.inf
        self.ac = self.actor_critic(self.env.observation_space, self.env.action_space, device=self.device, ngpu=self.ngpu, **self.ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        
        # Experience buffer
        self.replay_buffer = ReplayBuffer(int(self.replay_size))

        # Set up optimizers for actor and critic
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = Adam(chain(self.ac.q1.parameters(), self.ac.q2.parameters()), lr=self.q_lr)
                
    def update(self, experiences, timestep, update_policy=False):
        '''
        Do gradient updates for actor-critic models
        Args:
            experiences: sampled s, a, r, s', terminals from replay buffer.
            update_policy (bool): If True, update the actor network
        '''
        self.ac.train()
        self.ac_targ.train()
        # Get states, action, rewards, next_states, terminals from experiences
        states, actions, rewards, next_states, terminals = experiences
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        terminals = terminals.to(self.device)

        # --------------------- Optimizing critic ---------------------
        self.q_optimizer.zero_grad()
        # calculating q loss
        q1 = self.ac.q1(states, actions)
        q2 = self.ac.q2(states, actions)

        with torch.no_grad():
            # Trick 3: Target Policy Smoothing
            next_actions = self.ac_targ.pi(next_states)
            epsilon_noise = torch.randn_like(next_actions) * self.act_noise
            next_actions = torch.clamp(next_actions + epsilon_noise, -self.act_limit, self.act_limit)

            # Minimum target next_Q value
            next_q1 = self.ac_targ.q1(next_states, next_actions)
            next_q2 = self.ac_targ.q2(next_states, next_actions)
            next_Q = torch.min(next_q1, next_q2) * (1-terminals)
            Qprime = rewards + (self.gamma * next_Q)
        
        # MSE loss
        loss_q = ((q1-Qprime)**2).mean() + ((q2-Qprime)**2).mean()
        loss_info = dict(Q1vals=q1.detach().cpu().numpy().tolist(),
                        Q2Vals=q2.detach().cpu().numpy().tolist())

        loss_q.backward()
        self.q_optimizer.step()
        # Record loss q and loss pi and qvals in the form of loss_info
        self.logger.store(LossQ=loss_q.item(), **loss_info)
        self.tensorboard_logger.add_scalar("loss/q_loss", loss_q.item(), timestep)

        if update_policy:
            # --------------------- Optimizing actor ---------------------
            # Freeze Q-network so no computational resources is wasted in computing gradients
            for p in chain(self.ac.q1.parameters(), self.ac.q2.parameters()):
                p.requires_grad = False

            self.pi_optimizer.zero_grad()
            loss_pi = -self.ac.q1(states, self.ac.pi(states)).mean()
            loss_pi.backward()
            self.pi_optimizer.step()

            # Unfreeze Q-network for next update step
            for p in chain(self.ac.q1.parameters(), self.ac.q2.parameters()):
                p.requires_grad = True
                
            # Record loss q and loss pi and qvals in the form of loss_info
            self.logger.store(LossPi=loss_pi.item())
            self.tensorboard_logger.add_scalar("loss/pi_loss", loss_pi.item(), timestep)
            # update target networks
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    p_targ.data.mul_(self.tau)
                    p_targ.data.add_((1-self.tau)*p.data)

    def get_action(self, obs, noise_scale):
        '''
        Input the current observation into the actor network to calculate action to take.
        Args:
            obs (numpy ndarray): Current state of the environment. Only 1 state, not a batch of states
            noise_scale (float): Stddev for Gaussian exploration noise
        Return:
            Action (numpy ndarray): Scaled action that is clipped to environment's action limits
        '''
        self.ac.eval()
        self.ac_targ.eval()
        obs = torch.as_tensor([obs], dtype=torch.float32).to(self.device)
        action = self.ac.act(obs).squeeze()
        if len(action.shape) == 0:
            action = np.array([action])
        action += noise_scale*np.random.randn(self.act_dim)
        return np.clip(action, -self.act_limit, self.act_limit)

    def evaluate_agent(self):
        '''
        Run the current model through test environment for <self.num_test_episodes> episodes
        without noise exploration, and store the episode return and length into the logger.
        
        Used to measure how well the agent is doing.
        '''
        self.env.training = False
        for i in range(self.num_test_episodes):
            state, done, ep_ret, ep_len = self.env.reset(), False, 0, 0
            while not (done or (ep_len==self.max_ep_len)):
                # Take deterministic action with 0 noise added
                state, reward, done, _ = self.env.step(self.get_action(state, 0))
                ep_ret += reward
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        self.env.training = True

    def save_weights(self, best=False, fname=None):
        '''
        save the pytorch model weights of ac and ac_targ
        Args:

        '''
        if fname is not None:
            _fname = fname
        elif best:
            _fname = "best.pth"
        else:
            _fname = "model_weights.pth"

        print('saving checkpoint...')
        checkpoint = {
            'ac': self.ac.state_dict(),
            'ac_target': self.ac_targ.state_dict(),
            'pi_optimizer': self.pi_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(self.save_dir, _fname))
        self.replay_buffer.save(os.path.join(self.save_dir, "replay_buffer.pickle"))
        self.env.save(os.path.join(self.save_dir, "env.json"))
        print(f"checkpoint saved at {os.path.join(self.save_dir, _fname)}")

    def load_weights(self, best=True, load_buffer=True):
        '''
        Load the model weights and replay buffer from self.save_dir
        Args:
            best (bool): If True, save from the weights file with the best mean episode reward
            load_buffer (bool): If True, load the replay buffer from the pickled file
        '''
        if best:
            fname = "best.pth"
        else:
            fname = "model_weights.pth"
        checkpoint_path = os.path.join(self.save_dir, fname)
        if os.path.isfile(checkpoint_path):
            if load_buffer:
                self.replay_buffer.load(os.path.join(self.save_dir, "replay_buffer.pickle"))
            key = 'cuda' if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(checkpoint_path, map_location=key)
            self.ac.load_state_dict(sanitise_state_dict(checkpoint['ac'], self.ngpu>1))
            self.ac_targ.load_state_dict(sanitise_state_dict(checkpoint['ac_target'], self.ngpu>1))
            self.pi_optimizer.load_state_dict(sanitise_state_dict(checkpoint['pi_optimizer'], self.ngpu>1))
            self.q_optimizer.load_state_dict(sanitise_state_dict(checkpoint['q_optimizer'], self.ngpu>1))
            
            env_path = os.path.join(self.save_dir, "env.json")
            if os.path.isfile(env_path):
                self.env = self.env.load(env_path)
                print("Environment loaded")

            print('checkpoint loaded at {}'.format(checkpoint_path))
        else:
            raise OSError("Checkpoint file not found.")    

    def learn_one_trial(self, timesteps, trial_num):
        state, ep_ret, ep_len = self.env.reset(), 0, 0
        episode = 0
        for timestep in tqdm(range(timesteps)):
            # Until start_steps have elapsed, sample random actions from environment
            # to encourage more exploration, sample from policy network after that
            if timestep<=self.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.get_action(state, self.act_noise)

            # step the environment
            next_state, reward, done, _ = self.env.step(action)
            ep_ret += reward
            ep_len += 1

            # ignore the 'done' signal if it just times out after timestep>max_timesteps
            done = False if ep_len==self.max_ep_len else done

            # store experience to replay buffer
            self.replay_buffer.append(state, action, reward, next_state, done)

            # Critical step to update current state
            state = next_state
            
            # Update handling
            if timestep>=self.update_after and (timestep+1)%self.update_every==0:
                for j in range(self.update_every):
                    # Trick 2: “Delayed” Policy Updates.
                    update_policy = True if j%self.policy_delay==0 else False
                    experiences = self.replay_buffer.sample(self.batch_size)
                    self.update(experiences, timestep, update_policy=update_policy)
            
            # End of trajectory/episode handling
            if done or (ep_len==self.max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                # print(f"Episode reward: {ep_ret} | Episode Length: {ep_len}")
                state, ep_ret, ep_len = self.env.reset(), 0, 0
                episode += 1
                # Retrieve training reward
                x, y = self.logger.load_results(["EpLen", "EpRet"])
                self.tensorboard_logger.add_scalar('episodic_return_train', ep_ret, timestep)
                self.tensorboard_logger.flush()
                if len(x) > 0:
                    # Mean training reward over the last 50 episodes
                    mean_reward = np.mean(y[-50:])

                    # New best model
                    if mean_reward > self.best_mean_reward:
                        print("Num timesteps: {}".format(timestep))
                        print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                        self.best_mean_reward = mean_reward
                        self.save_weights(fname=f"best_{trial_num}.pth")
                    
                    if self.env.spec.reward_threshold is not None and self.best_mean_reward >= self.env.spec.reward_threshold:
                        print("Solved Environment, stopping iteration...")
                        return

                # self.evaluate_agent()
                self.logger.dump()

    def learn(self, timesteps, num_trials=1):
        '''
        Function to learn using TD3.
        Args:
            timesteps (int): number of timesteps to train for
            num_trials (int): Number of times to train the agent
        '''
        self.env.training = True
        best_reward_trial = -np.inf
        for trial in range(num_trials):
            self.tensorboard_logger = SummaryWriter(log_dir=os.path.join(self.tensorboard_logdir, f'{trial+1}'))
            self.learn_one_trial(timesteps, trial+1)

            if self.best_mean_reward > best_reward_trial:
                best_reward_trial = self.best_mean_reward
                self.save_weights(best=True)

            self.logger.reset()
            self.reinit_network()
            print()
            print(f"Trial {trial+1}/{num_trials} complete")
            
    def test(self, timesteps=None, render=False, record=False):
        '''
        Test the agent in the environment
        Args:
            render (bool): If true, render the image out for user to see in real time
            record (bool): If true, save the recording into a .gif file at the end of episode
            timesteps (int): number of timesteps to run the environment for. Default None will run to completion
        Return:
            Ep_Ret (int): Total reward from the episode
            Ep_Len (int): Total length of the episode in terms of timesteps
        '''
        self.env.training = False
        if render:
            self.env.render('human')
        state, done, ep_ret, ep_len = self.env.reset(), False, 0, 0
        img = []
        if record:
            img.append(self.env.render('rgb_array'))

        if timesteps is not None:
            for i in range(timesteps):
                # Take deterministic action with 0 noise added
                state, reward, done, _ = self.env.step(self.get_action(state, 0))
                if record:
                    img.append(self.env.render('rgb_array'))
                else:
                    self.env.render()
                ep_ret += reward
                ep_len += 1                
        else:
            while not (done or (ep_len==self.max_ep_len)):
                # Take deterministic action with 0 noise added
                state, reward, done, _ = self.env.step(self.get_action(state, 0))
                if record:
                    img.append(self.env.render('rgb_array'))
                else:
                    self.env.render()
                ep_ret += reward
                ep_len += 1

        if record:
            imageio.mimsave(f'{os.path.join(self.save_dir, "recording.gif")}', [np.array(img) for i, img in enumerate(img) if i%2 == 0], fps=29)

        self.env.training = True
        return ep_ret, ep_len      

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPoleContinuousBulletEnv-v0', help='environment_id')
    parser.add_argument('--config_path', type=str, default='td3_config.json', help='path to config.json')
    parser.add_argument('--timesteps', type=int, required=True, help='specify number of timesteps to train for') 
    parser.add_argument('--seed', type=int, default=0, help='seed number for reproducibility')
    return parser.parse_args()

def main():
    args = parse_arguments()
    save_dir = os.path.join("Model_Weights", args.env, "td3")
    logger_kwargs = {
        "output_dir": save_dir
    }
    with open(args.config_path) as f:
        model_kwargs = json.load(f)

    model = TD3(lambda: gym.make(args.env), save_dir, seed=args.seed, logger_kwargs=logger_kwargs, **model_kwargs)
    model.learn(args.timesteps)

if __name__ == '__main__':
    main()