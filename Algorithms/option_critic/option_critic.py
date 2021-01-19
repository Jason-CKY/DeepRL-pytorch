import torch
import numpy as np
import os
import imageio

from math import exp
from Algorithms.utils import to_tensor, sanitise_state_dict, get_actor_critic_module
from Algorithms.option_critic.replay_buffer import ReplayBuffer
from Logger.logger import Logger
from copy import deepcopy
from torch.optim import Adam
from tqdm import tqdm
from gym.spaces import Discrete
from itertools import chain

class OptionCritic:
    def __init__(self, env_fn, save_dir, oc_kwargs=dict(), seed=0, optimizer=Adam,
         replay_size=int(1e6), gamma=0.99, eps_start=1.0, eps_end=0.1, eps_decay=20000,
         lr=1e-3, batch_size=100, update_frequency=4, termination_reg=0.01, entropy_reg=0.2, load_path=None,
         max_ep_len=1000, freeze_interval=200, logger_kwargs=dict(), save_freq=1, ngpu=1):    
        '''
        Option-Critic Architecture https://arxiv.org/abs/1609.05140
        Args:
            env_fn: function to create the gym environment
            save_dir: path to save directory
            actor_critic: Class for the actor-critic pytorch module
            oc_kwargs (dict): any keyword argument for the option_critic
                        num_options (int): Number of options for the option-critic architecture
                        vae_weights_path (str): Path to the pretrained VAE weights file
                        hidden_sizes (List): number of neurons in hidden layer                        
            seed (int): seed for random generators
            replay_size (int): Maximum length of replay buffer.
            gamma (float): Discount factor. (Always between 0 and 1.)
            eps_start (float): Starting value for epsilon (used in epsilon greedy policy over options)
            eps_end (float): minimum value for epsilon
            eps_decay (int): number of timesteps to decay eps from eps_start to eps_end
            lr (float): Learning rate for OptionCritic as they share parameters
            batch_size (int): Batch size for learning
            update_frequency (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long 
                you wait between updates, the ratio of env steps to gradient steps 
                is locked to 1.
            termination_reg (float): Regularization term to decrease termination probability
            entropy_reg (float): Entropy regularization coefficient. (Equivalent to 
                                inverse of reward scale in the original SAC paper.)
            polyak (float): Interpolation factor in polyak averaging for target 
                            networks.
            max_ep_len (int): Maximum length of trajectory / episode / rollout.
            logger_kwargs (dict): Keyword args for Logger. 
                        (1) output_dir = None
                        (2) output_fname = 'progress.pickle'
            save_freq (int): How often (in terms of gap between episodes) to save
                the current policy and value function.
        '''
        # logger stuff
        self.logger = Logger(**logger_kwargs)

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.env = env_fn()

        # Create actor-critic module
        self.ngpu = ngpu
        self.oc_kwargs = oc_kwargs
        self.option_critic = get_actor_critic_module(self.oc_kwargs, 'option_critic')
        self.oc = self.option_critic(self.env.observation_space, self.env.action_space, device=self.device, ngpu=self.ngpu, **oc_kwargs)
        if load_path is not None:
            self.load_weights(load_path)

        self.oc_targ = deepcopy(self.oc)
        # Freeze target networks with respect to optimizers
        for p in self.oc_targ.parameters():
            p.requires_grad = False
        
        # Experience buffer
        self.replay_size = replay_size
        self.replay_buffer = ReplayBuffer(int(replay_size))

        # Set up optimizers for actor and critic
        self.optimizer_class = optimizer
        self.lr = lr
        self.optimizer = self.optimizer_class(self.oc.parameters(), lr=self.lr)
        
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.freeze_interval = freeze_interval
        self.termination_reg = termination_reg
        self.max_ep_len = self.env.spec.max_episode_steps if self.env.spec.max_episode_steps is not None else max_ep_len
        # self.polyak = polyak
        self.entropy_reg = entropy_reg
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.best_mean_reward = -np.inf

    def reinit_network(self):
        '''
        Re-initialize network weights and optimizers for a fresh agent to train
        '''        
        
        # Create actor-critic module
        self.best_mean_reward = -np.inf
        self.oc = self.option_critic(self.env.observation_space, self.env.action_space, device=self.device, ngpu=self.ngpu, **self.oc_kwargs)
        self.oc_targ = deepcopy(self.oc)

        # Freeze target networks with respect to optimizers
        for p in self.oc_targ.parameters():
            p.requires_grad = False
        
        # Experience buffer
        self.replay_buffer = ReplayBuffer(int(self.replay_size))

        # Set up optimizers for option_critic
        self.optimizer = self.optimizer_class(self.oc.parameters(), lr=self.lr)
   
    def get_critic_loss(self, experiences):
        '''
        Do gradient updates for actor-critic models
        Args:
            experiences: sampled s, w, r, s', terminals from replay buffer.
        '''
        # Get states, action, rewards, next_states, terminals from experiences
        obs, options, rewards, next_obs, done = experiences
        batch_idx = torch.arange(len(options)).long()
        obs = obs.to(self.device)
        options = options.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        # The loss is the TD loss of Q and the update target, so we need to calculate Q
        states = self.oc.encode_state(to_tensor(obs))
        Q      = self.oc.get_Q(states)
        
        # the update target contains Q_next, but for stable learning we use prime network for this
        next_states_prime = self.oc_targ.encode_state(to_tensor(next_obs))
        next_Q_prime      = self.oc_targ.get_Q(next_states_prime) # detach?

        # Additionally, we need the beta probabilities of the next state
        next_states            = self.oc.encode_state(to_tensor(next_obs))
        next_termination_probs = self.oc.get_terminations(next_states).detach()
        next_options_term_prob = next_termination_probs[batch_idx, options]

        # Now we can calculate the update target gt
        gt = rewards + (1-done) * self.gamma * \
            ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + next_options_term_prob  * next_Q_prime.max(dim=-1)[0])

        # to update Q we want to use the actual network, not the prime
        td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
        self.logger.store(loss_Q=td_err)
        return td_err

    def get_pi_loss(self, obs, option, logp, entropy, reward, done, next_obs):
        state = self.oc.encode_state(to_tensor(obs))
        next_state = self.oc.encode_state(to_tensor(next_obs))
        next_state_prime = self.oc_targ.encode_state(to_tensor(next_obs))

        option_term_prob = self.oc.get_terminations(state)[option]
        next_option_term_prob = self.oc.get_terminations(next_state)[option].detach()

        Q = self.oc.get_Q(state).detach().squeeze()
        next_Q_prime = self.oc_targ.get_Q(next_state_prime).detach().squeeze()

        # Target update gt
        gt = reward + (1 - done) * self.gamma * \
            ((1 - next_option_term_prob) * next_Q_prime[option] + next_option_term_prob  * next_Q_prime.max(dim=-1)[0])

        # The termination loss
        termination_loss = option_term_prob * (Q[option] - Q.max(dim=-1)[0] + self.termination_reg) * (1 - done)
        termination_loss = termination_loss.mean()
        
        # actor-critic policy gradient with entropy regularization
        policy_loss = -logp * (gt.detach() - Q[option]) 
        policy_loss = policy_loss - self.entropy_reg * entropy if isinstance(self.env.action_space, Discrete) else policy_loss
        policy_loss = policy_loss.mean()
        loss = policy_loss + termination_loss
        self.logger.store(termination_loss=termination_loss.item(), policy_loss=policy_loss.item())

        return loss

    def evaluate_agent(self):
        '''
        Run the current model through test environment for <self.num_test_episodes> episodes
        without noise exploration, and store the episode return and length into the logger.
        
        Used to measure how well the agent is doing.
        '''
        self.env.training=False
        for i in range(self.num_test_episodes):
            state, done, ep_ret, ep_len = self.env.reset(), False, 0, 0
            while not (done or (ep_len==self.max_ep_len)):
                # Take deterministic action with 0 noise added
                state, reward, done, _ = self.env.step(self.oc.get_action(state, 0))
                ep_ret += reward
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        self.env.training=True

    def save_weights(self, best=False, fname=None):
        '''
        save the pytorch model weights of ac and oc_targ
        as well as pickling the environment to preserve any env parameters like normalisation params
        Args:
            best(bool): if true, save it as best.pth
            fname(string): if specified, save it as <fname>
        '''
        if fname is not None:
            _fname = fname
        elif best:
            _fname = "best.pth"
        else:
            _fname = "model_weights.pth"
        
        print('saving checkpoint...')
        checkpoint = {
            'oc': self.oc.state_dict(),
            'oc_target': self.oc_targ.state_dict()
        }
        torch.save(checkpoint, os.path.join(self.save_dir, _fname))
        self.replay_buffer.save(os.path.join(self.save_dir, "replay_buffer.pickle"))
        self.env.save(os.path.join(self.save_dir, "env.json"))
        print(f"checkpoint saved at {os.path.join(self.save_dir, _fname)}")

    def load_weights(self, path=None, best=True, load_buffer=True):
        '''
        Load the model weights and replay buffer from self.save_dir
        Args:
            best (bool): If True, save from the weights file with the best mean episode reward
            load_buffer (bool): If True, load the replay buffer from the pickled file
        '''
        if path is None:
            if best:
                fname = "best.pth"
            else:
                fname = "model_weights.pth"
            checkpoint_path = os.path.join(self.save_dir, fname)
        else:
            checkpoint_path = path

        if os.path.isfile(checkpoint_path):
            if load_buffer:
                self.replay_buffer.load(os.path.join(self.save_dir, "replay_buffer.pickle"))
            key = 'cuda' if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(checkpoint_path, map_location=key)
            self.oc.load_state_dict(sanitise_state_dict(checkpoint['oc'], self.ngpu>1))
            self.oc_targ.load_state_dict(sanitise_state_dict(checkpoint['oc_target'], self.ngpu>1))

            env_path = os.path.join(self.save_dir, "env.json")
            if os.path.isfile(env_path):
                self.env = self.env.load(env_path)
                print("Environment loaded")
            
            print('checkpoint loaded at {}'.format(checkpoint_path))
        else:
            raise OSError("Checkpoint file not found.")    

    def update_epsilon(self, eps, timestep):
        eps = self.eps_end + (self.eps_start - self.eps_end) * exp(-timestep / self.eps_decay)
        return eps

    def learn_one_trial(self, timesteps, trial_num):
        self.oc.train(); self.oc_targ.train()
        epsilon = self.eps_start
        obs, ep_ret, ep_len, curr_op_len, option_termination = self.env.reset(), 0, 0, 0, True
        current_option = 0
        episode = 0
        option_lengths = {opt:[] for opt in range(self.oc.num_options)}

        for timestep in tqdm(range(1, timesteps+1)):
            epsilon = self.update_epsilon(epsilon, timestep)
            
            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = self.oc.get_option(to_tensor(obs), epsilon)
                curr_op_len = 0
            
            # Until start_steps have elapsed, sample random actions from environment
            # to encourage more exploration, sample from policy network after that
            action, logp, entropy = self.oc.get_action(to_tensor(obs), current_option)

            # step the environment
            next_obs, reward, done, _ = self.env.step(action)
            ep_ret += reward
            ep_len += 1

            # ignore the 'done' signal if it just times out after timestep>max_timesteps
            done = False if ep_len==self.max_ep_len else done

            # store experience to replay buffer
            self.replay_buffer.append(obs, current_option, reward, next_obs, done)

            # Critical step to update current state
            obs = next_obs
            option_termination = self.oc.predict_option_termination(to_tensor(obs), current_option)

            if self.replay_buffer.size() >= self.batch_size:
                loss = self.get_pi_loss(obs, current_option, logp, entropy, reward, done, next_obs)
                if timestep%self.update_frequency==0:
                    experiences = self.replay_buffer.sample(self.batch_size)
                    loss += self.get_critic_loss(experiences)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            if timestep % self.freeze_interval == 0:
                self.oc_targ.load_state_dict(self.oc.state_dict())
            
            # End of trajectory/episode handling
            if done or (ep_len==self.max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len, OptLen=option_lengths)
                # print(f"Episode reward: {ep_ret} | Episode Length: {ep_len}")
                obs, ep_ret, ep_len = self.env.reset(), 0, 0
                option_lengths = {opt:[] for opt in range(self.oc.num_options)}
                episode += 1
                # Retrieve training reward
                x, y = self.logger.load_results(["EpLen", "EpRet"])
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
        Function to learn using DDPG.
        Args:
            timesteps (int): number of timesteps to train for
        '''
        self.env.training=True
        best_reward_trial = -np.inf
        for trial in range(num_trials):
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
        self.oc.eval(); self.oc_targ.eval()
        self.env.training=False
        if render:
            self.env.render('human')
        obs, ep_ret, ep_len, option_termination = self.env.reset(), 0, 0, True
        done = False
        current_option = 0
        img = []
        if record:
            img.append(self.env.render('rgb_array'))

        if timesteps is not None:
            for i in range(timesteps):
                if option_termination:
                    current_option = self.oc.get_option(obs, 0, greedy=True)
                obs, reward, done, _ = self.env.step(self.oc.get_action(obs, current_option))
                option_termination = self.oc.predict_option_termination(to_tensor(obs), current_option)
                if record:
                    img.append(self.env.render('rgb_array'))
                else:
                    self.env.render()
                ep_ret += reward
                ep_len += 1                
        else:
            while not (done or (ep_len==self.max_ep_len)):
                if option_termination:
                    current_option = self.oc.get_option(obs, 0, greedy=True)
                obs, reward, done, _ = self.env.step(self.oc.get_action(obs, current_option))
                option_termination = self.oc.predict_option_termination(to_tensor(obs), current_option)
                if record:
                    img.append(self.env.render('rgb_array'))
                else:
                    self.env.render()
                ep_ret += reward
                ep_len += 1

        if record:
            imageio.mimsave(f'{os.path.join(self.save_dir, "recording.gif")}', [np.array(img) for i, img in enumerate(img) if i%2 == 0], fps=29)

        self.env.training=True
        return ep_ret, ep_len      
