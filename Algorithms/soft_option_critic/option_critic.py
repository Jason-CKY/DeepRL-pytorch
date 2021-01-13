import gym
import pybullet_envs
import torch
import numpy as np
import time
import os
import imageio

from torch.distributions import Categorical
from Wrappers.normalize_observation import Normalize_Observation
from Algorithms.soft_option_critic.core import OptionCriticVAE
from Algorithms.utils import to_tensor, sanitise_state_dict
from Algorithms.soft_option_critic.replay_buffer import ReplayBuffer
from Logger.logger import Logger
from copy import deepcopy
from torch.optim import Adam, RMSprop
from tqdm import tqdm
from itertools import chain

class OptionCritic:
    def __init__(self, env_fn, save_dir, oc_kwargs=dict(), seed=0, optimizer=Adam,
         replay_size=int(1e6), gamma=0.99, lr=dict(), batch_size=128, update_after=int(1e4), 
         update_every=1, alpha=1.0, polyak=0.995, max_ep_len=1000, logger_kwargs=dict(), 
         lambda_1=1, lambda_2=5, save_freq=1, ngpu=1):    
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
            lr (dict): Learning rate for OptionCritic as they share parameters
                        Q (float): Learning rate for Q-Values
                        U (float): Learning rate for U-Values
                        pi_option (float): Learning rate for High level policies    
                        pi_action (float): Learning rate for Low level policies
            batch_size (int): Batch size for learning
            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.
            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long 
                you wait between updates, the ratio of env steps to gradient steps 
                is locked to 1.
            alpha (float): Entropy regularization
                                alpha -> infinity: uniform policy;
                                alpha -> 0: greedy policy
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

        # Action Limit for clamping
        self.act_limit = self.env.action_space.high[0]
        self.act_dim = self.env.action_space.shape[0]

        # Create actor-critic module
        self.ngpu = ngpu
        self.option_critic = OptionCriticVAE
        self.oc_kwargs = oc_kwargs
        self.oc = self.option_critic(self.env.observation_space, self.env.action_space, device=self.device, ngpu=self.ngpu, **oc_kwargs)
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
       
        self.optimizers = {
            'Q': self.optimizer_class(chain(self.oc.encoder.parameters(), 
                    self.oc.Q1.parameters(), self.oc.Q2.parameters()), lr=self.lr['Q']),
            'U': self.optimizer_class(chain(self.oc.encoder.parameters(), 
                    self.oc.U1.parameters(), self.oc.U2.parameters()), lr=self.lr['U']),
            'pi_option': self.optimizer_class(chain(self.oc.encoder.parameters(), self.oc.pi_high.parameters()), 
                                            lr=self.lr['pi_option']),
            'pi_action': self.optimizer_class(chain(self.oc.encoder.parameters(), self.oc.pi_low.parameters()), 
                                            lr=self.lr['pi_action'])
        }

        self.gamma = gamma
        self.batch_size = batch_size
        self.update_after = update_after
        self.update_every = update_every
        self.mutual_info_weight = lambda_1
        self.noise_influence_weight = lambda_2

        self.alpha = alpha
        self.max_ep_len = self.env.spec.max_episode_steps if self.env.spec.max_episode_steps is not None else max_ep_len
        self.polyak = polyak
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
        self.optimizers = {
            'Q': self.optimizer_class(chain(self.oc.encoder.parameters(), 
                    self.oc.Q1.parameters(), self.oc.Q2.parameters()), lr=self.lr['Q']),
            'U': self.optimizer_class(chain(self.oc.encoder.parameters(), 
                    self.oc.U1.parameters(), self.oc.U2.parameters()), lr=self.lr['U']),
            'pi_option': self.optimizer_class(chain(self.oc.encoder.parameters(), self.oc.pi_high.parameters()), 
                                            lr=self.lr['pi_option']),
            'pi_action': self.optimizer_class(chain(self.oc.encoder.parameters(), self.oc.pi_low.parameters()), 
                                            lr=self.lr['pi_action'])
        }
    
    def update_target_network(self):
        with torch.no_grad():
            for p, p_targ in zip(self.oc.parameters(), self.oc_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def update(self, experiences):
        '''
        Do gradient updates for actor-critic models
        Args:
            experiences: sampled s, a, r, s', terminals from replay buffer.
        '''
        # Get states, action, rewards, next_states, terminals from experiences
        self.oc.train()
        self.oc_targ.train()
        
        prev_options, obs, options, actions, rewards, next_obs, logp_actions = experiences
        prev_options = prev_options.to(self.device)
        obs = obs.to(self.device)
        options = options.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        logp_actions = logp_actions.to(self.device)
        # ------------------ TODO: optimizing U-value functions ------------------------------
        p_next_options = self.oc_targ.pi_high(next_obs, options)
        next_options = Categorical(probs=p_next_options).sample()

        MI = self.mutual_info(options, obs, actions)    # TODO
        TV_distance = self.TV_distance()                # TODO
        logp = self.get_logp_options() # logp(z | s, a)
        V_next = torch.min(self.oc_targ.get_U1(next_obs, next_options), self.oc_targ.get_U2(next_obs, next_options)) - self.alpha*torch.log(p_next_options)
        Q_estimation = rewards + self.alpha*(self.mutual_info_weight*MI - self.noise_influence_weight*TV_distance - logp) \
                                + self.gamma*V_next

        loss_Q1 = 0.5*((self.oc.Q1(obs, options, actions) - Q_estimation)**2).mean()
        loss_Q2 = 0.5*((self.oc.Q2(obs, options, actions) - Q_estimation)**2).mean()
        loss_Q = loss_Q1 + loss_Q2
        self.optimizers['Q'].zero_grad()
        loss_Q.backward()
        self.optimizers['Q'].step()
        

        # ------------------ optimizing U-value functions ------------------------------
        Q1_current = self.oc.get_Q1(obs, options, actions)
        Q2_current = self.oc.get_Q2(obs, options, actions)
        # logp_actions = self.oc.get_action_logp(obs, options, actions)
        loss_U1 = 0.5*((self.oc.get_U1(obs, options) - (torch.min(Q1_current, 
                    Q2_current) - self.alpha*logp_actions))**2).mean()
        loss_U2 = 0.5*((self.oc.get_U2(obs, options) - (torch.min(Q1_current, 
                    Q2_current) - self.alpha*logp_actions))**2).mean()

        loss_U = loss_U1 + loss_U2
        self.optimizers['U'].zero_grad()
        loss_U.backward()
        self.optimizers['U'].step()

        # --------------------- optimizing high level policy (option-selection) ----------------------
        p_options = self.oc.pi_high(self.oc.encode_state(obs), prev_options) # get probabilities for every single option
        U1_all_options = torch.stack([self.oc.get_U1(obs, torch.tensor([option]*obs.shape[0]).to(self.device)).squeeze() for option in range(self.oc_kwargs['num_options'])], dim=1) # shape should be: [batch_size, num_options]
        U2_all_options = torch.stack([self.oc.get_U2(obs, torch.tensor([option]*obs.shape[0]).to(self.device)).squeeze() for option in range(self.oc_kwargs['num_options'])], dim=1)
        
        loss_pi_options = p_options.T @ (self.alpha*torch.log(p_options) - torch.min(U1_all_options, U2_all_options))
        loss_pi_options = loss_pi_options.mean()
        self.optimizers['pi_option'].zero_grad()
        loss_pi_options.backward()
        self.optimizers['pi_option'].step()

        # ------------------ optimizing low level policy -----------------------------
        sample_actions, log_p = self.oc.act(obs, options) 
        sample_actions = to_tensor(sample_actions).to(self.device)
        loss_pi_actions = self.alpha*log_p - torch.min(self.oc.get_Q1(obs, options, sample_actions),
                        self.oc.get_Q2(obs, options, sample_actions))
        loss_pi_actions = loss_pi_actions.mean()
        self.optimizers['pi_action'].zero_grad()
        loss_pi_actions.backward()
        self.optimizers['pi_action'].step()

        print(loss_U)
        print(loss_pi_options)
        print(loss_pi_actions)
        if torch.isnan(loss_pi_actions):
            print(sample_actions)
            print(log_p)
        # Record loss q and loss pi and qvals in the form of loss_info
        # self.logger.store(loss_q1=loss_q1.item(), loss_q2=loss_q2.item, loss_q = loss_q.item(),
        #                     policy_loss=policy_loss.item(), termination_loss=termination_loss.item())

        self.update_target_network()

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
            self.oc.load_state_dict(sanitise_state_dict(checkpoint['oc'], self.ngpu>1))
            self.oc_targ.load_state_dict(sanitise_state_dict(checkpoint['oc_target'], self.ngpu>1))

            env_path = os.path.join(self.save_dir, "env.json")
            if os.path.isfile(env_path):
                self.env = self.env.load(env_path)
                print("Environment loaded")
            
            print('checkpoint loaded at {}'.format(checkpoint_path))
        else:
            raise OSError("Checkpoint file not found.")    

    def learn_one_trial(self, timesteps, trial_num):
        self.oc.train(); self.oc_targ.train()
        obs, ep_ret, ep_len, curr_op_len = self.env.reset(), 0, 0, 0
        prev_option = 0
        episode = 0
        option_lengths = {opt:[] for opt in range(self.oc.num_options)}

        for timestep in tqdm(range(1, timesteps+1)): 
            current_option = self.oc.get_option(to_tensor(obs), prev_option)
            if current_option != prev_option:
                option_lengths[prev_option].append(curr_op_len)
                curr_op_len = 0
            
            # Until start_steps have elapsed, sample random actions from environment
            # to encourage more exploration, sample from policy network after that
            action, logp_actions = self.oc.act(to_tensor(obs), current_option)

            # step the environment
            next_obs, reward, done, _ = self.env.step(action)
            ep_ret += reward
            ep_len += 1
            curr_op_len += 1

            # ignore the 'done' signal if it just times out after timestep>max_timesteps
            done = False if ep_len==self.max_ep_len else done

            # store experience to replay buffer
            self.replay_buffer.append(prev_option, obs, current_option, action, reward, next_obs, logp_actions.detach().cpu().numpy())
            prev_option = current_option

            # Critical step to update current state
            obs = next_obs

            if self.replay_buffer.size() >= self.batch_size and timestep%self.update_every==0 \
                and timestep > self.update_after:
                for _ in range(self.update_every):
                    experiences = self.replay_buffer.sample(self.batch_size)
                    self.update(experiences)
                # self.save_weights()
            # End of trajectory/episode handling
            if done or (ep_len==self.max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len, OptLen=option_lengths)
                print(f"Episode reward: {ep_ret} | Episode Length: {ep_len}")
                state, ep_ret, ep_len = self.env.reset(), 0, 0
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
        state, done, ep_ret, ep_len = self.env.reset(), False, 0, 0
        prev_option = 0
        img = []
        if record:
            img.append(self.env.render('rgb_array'))

        if timesteps is not None:
            for i in range(timesteps):
                current_option = self.oc.get_option(to_tensor(obs), prev_option, deterministic=True)
                action, _ = self.oc.act(to_tensor(obs), current_option, deterministic=True)
                prev_option = current_option
                state, reward, done, _ = self.env.step(action)
                if record:
                    img.append(self.env.render('rgb_array'))
                else:
                    self.env.render()
                ep_ret += reward
                ep_len += 1                
        else:
            while not (done or (ep_len==self.max_ep_len)):
                current_option = self.oc.get_option(to_tensor(obs), prev_option, deterministic=True)
                action, _ = self.oc.act(to_tensor(obs), current_option, deterministic=True)
                prev_option = current_option
                state, reward, done, _ = self.env.step(action)
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
