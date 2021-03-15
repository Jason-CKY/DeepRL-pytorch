import gym
import pybullet_envs
import torch
import numpy as np
import time
import argparse
import os
import imageio

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from Wrappers.normalize_observation import Normalize_Observation
from Algorithms.ppo.core import MLPActorCritic, CNNActorCritic
from Algorithms.utils import get_actor_critic_module, sanitise_state_dict
from Algorithms.ppo.gae_buffer import GAEBuffer
from Logger.logger import Logger
from copy import deepcopy
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class PPO:
    def __init__(self, env_fn, save_dir, ac_kwargs=dict(), seed=0, tensorboard_logdir = None,
         steps_per_epoch=400, batch_size=400, gamma=0.99, clip_ratio=0.2, 
         vf_lr=1e-3, pi_lr=3e-4, train_v_iters=80, train_pi_iters=80, 
         lam=0.97, max_ep_len=1000, target_kl=0.01, logger_kwargs=dict(), save_freq=10, ngpu=1):
        """
        Proximal Policy Optimization 
        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.
            save_dir: path to save directory
            actor_critic: Class for the actor-critic pytorch module
            ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
                function you provided to TRPO.
            seed (int): Seed for random number generators.
            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
                for the agent and the environment in each epoch.
            batch_size (int): The buffer is split into batches of batch_size to learn from
            gamma (float): Discount factor. (Always between 0 and 1.)
            clip_ratio (float): Hyperparameter for clipping in the policy objective.
                Roughly: how far can the new policy go from the old policy while 
                still profiting (improving the objective function)? The new policy 
                can still go farther than the clip_ratio says, but it doesn't help
                on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
                denoted by :math:`\epsilon`. 
            pi_lr (float): Learning rate for policy optimizer.
            vf_lr (float): Learning rate for value function optimizer.
            train_v_iters (int): Number of gradient descent steps to take on 
                value function per epoch.
            train_pi_iters (int): Maximum number of gradient descent steps to take 
                on policy loss per epoch. (Early stopping may cause optimizer
                to take fewer than this.)    
            lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
                close to 1.)
            max_ep_len (int): Maximum length of trajectory / episode / rollout.
            target_kl (float): Roughly what KL divergence we think is appropriate
                between new and old policies after an update. This will get used 
                for early stopping. (Usually small, 0.01 or 0.05.)
            logger_kwargs (dict): Keyword args for Logger. 
                            (1) output_dir = None
                            (2) output_fname = 'progress.pickle'
            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.
        """
        # logger stuff
        self.logger = Logger(**logger_kwargs)

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = env_fn()
        self.vf_lr = vf_lr
        self.pi_lr = pi_lr
        self.steps_per_epoch = steps_per_epoch # if steps_per_epoch > self.env.spec.max_episode_steps else self.env.spec.max_episode_steps
        
        self.max_ep_len = max_ep_len
        # self.max_ep_len = self.env.spec.max_episode_steps if self.env.spec.max_episode_steps is not None else max_ep_len
        self.train_v_iters = train_v_iters
        self.train_pi_iters = train_pi_iters

        # Main network
        self.ngpu = ngpu
        self.actor_critic = get_actor_critic_module(ac_kwargs, 'ppo')
        self.ac_kwargs = ac_kwargs
        self.ac = self.actor_critic(self.env.observation_space, self.env.action_space, device=self.device, ngpu=self.ngpu, **ac_kwargs)

        # Create Optimizers
        self.v_optimizer = optim.Adam(self.ac.v.parameters(), lr=self.vf_lr)
        self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=self.pi_lr)

        # GAE buffer
        self.gamma = gamma
        self.lam = lam
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        self.buffer = GAEBuffer(self.obs_dim, self.act_dim, self.steps_per_epoch, self.device, self.gamma, self.lam)
        self.batch_size = batch_size

        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.best_mean_reward = -np.inf
        self.save_dir = save_dir
        self.save_freq = save_freq

        self.tensorboard_logdir = tensorboard_logdir

    def reinit_network(self):
        '''
        Re-initialize network weights and optimizers for a fresh agent to train
        '''

        # Main network
        self.best_mean_reward = -np.inf
        self.ac = self.actor_critic(self.env.observation_space, self.env.action_space, device=self.device, ngpu=self.ngpu, **self.ac_kwargs)

        # Create Optimizers
        self.v_optimizer = optim.Adam(self.ac.v.parameters(), lr=self.vf_lr)
        self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=self.pi_lr)

        self.buffer = GAEBuffer(self.obs_dim, self.act_dim, self.steps_per_epoch, self.device, self.gamma, self.lam)


    def update(self):
        self.ac.train()
        data = self.buffer.get()
        obs_ = data['obs']
        act_ = data['act']
        ret_ = data['ret']
        adv_ = data['adv']
        logp_old_ = data['logp']

        for index in BatchSampler(SubsetRandomSampler(range(self.steps_per_epoch)), self.batch_size, False):
            obs = obs_[index]
            act = act_[index]
            ret = ret_[index]
            adv = adv_[index]
            logp_old = logp_old_[index]

            # ---------------------Recording the losses before the updates --------------------------------
            pi, logp = self.ac.pi(obs, act)
            ratio = torch.exp(logp - logp_old)
            clipped_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
            loss_pi = -torch.min(ratio*adv, clipped_adv).mean()
            v = self.ac.v(obs)
            loss_v = ((v-ret)**2).mean()

            self.logger.store(LossV=loss_v.item(), LossPi=loss_pi.item())
            # --------------------------------------------------------------------------------------------   
            
            # Update Policy     
            for i in range(self.train_pi_iters):
                # Policy loss
                self.pi_optimizer.zero_grad()
                pi, logp = self.ac.pi(obs, act)
                ratio = torch.exp(logp - logp_old)
                clipped_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
                loss_pi = -torch.min(ratio*adv, clipped_adv).mean()
                approx_kl = (logp - logp_old).mean().item()
                if approx_kl > 1.5*self.target_kl:
                    print(f"Early stopping at step {i} due to reaching max kl")
                    break
                loss_pi.backward()
                self.pi_optimizer.step()

            # Update Value Function
            for _ in range(self.train_v_iters):
                self.v_optimizer.zero_grad()
                v = self.ac.v(obs)
                loss_v = ((v-ret)**2).mean()
                loss_v.backward()
                self.v_optimizer.step()

    def save_weights(self, best=False, fname=None):
        '''
        save the pytorch model weights of critic and actor networks
        '''
        if fname is not None:
            _fname = fname
        elif best:
            _fname = "best.pth"
        else:
            _fname = "model_weights.pth"

        print('saving checkpoint...')
        checkpoint = {
            'v': self.ac.v.state_dict(),
            'pi': self.ac.pi.state_dict(),
            'v_optimizer': self.v_optimizer.state_dict(),
            'pi_optimizer': self.pi_optimizer.state_dict()
        }
        self.env.save(os.path.join(self.save_dir, "env.json"))
        torch.save(checkpoint, os.path.join(self.save_dir, _fname))
        print(f"checkpoint saved at {os.path.join(self.save_dir, _fname)}")

    def load_weights(self, best=True):
        '''
        Load the model weights and replay buffer from self.save_dir
        Args:
            best (bool): If True, save from the weights file with the best mean episode reward
        '''
        if best:
            fname = "best.pth"
        else:
            fname = "model_weights.pth"
        checkpoint_path = os.path.join(self.save_dir, fname)
        if os.path.isfile(checkpoint_path):
            key = 'cuda' if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(checkpoint_path, map_location=key)
            self.ac.v.load_state_dict(sanitise_state_dict(checkpoint['v'], self.ngpu>1))
            self.ac.pi.load_state_dict(sanitise_state_dict(checkpoint['pi'], self.ngpu>1))
            self.v_optimizer.load_state_dict(sanitise_state_dict(checkpoint['v_optimizer'], self.ngpu>1))
            self.pi_optimizer.load_state_dict(sanitise_state_dict(checkpoint['pi_optimizer'], self.ngpu>1))

            env_path = os.path.join(self.save_dir, "env.json")
            if os.path.isfile(env_path):
                self.env = self.env.load(env_path)
                print("Environment loaded")
            print('checkpoint loaded at {}'.format(checkpoint_path))
        else:
            raise OSError("Checkpoint file not found.")    

    def learn_one_trial(self, timesteps, trial_num):
        ep_rets = []
        epochs = int((timesteps/self.steps_per_epoch) + 0.5)
        print("Rounded off to {} epochs with {} steps per epoch, total {} timesteps".format(epochs, self.steps_per_epoch, epochs*self.steps_per_epoch))
        start_time = time.time()
        obs, ep_ret, ep_len = self.env.reset(), 0, 0
        ep_num = 0
        for epoch in tqdm(range(epochs)):
            for t in range(self.steps_per_epoch):
                # step the environment
                a, v, logp = self.ac.step(torch.as_tensor(obs, dtype=torch.float32).to(self.device))
                next_obs, reward, done, _ = self.env.step(a)
                ep_ret += reward
                ep_len += 1
                
                # Add experience to buffer
                self.buffer.store(obs, a, reward, v, logp)

                obs = next_obs
                timeout = ep_len == self.max_ep_len
                terminal = done or timeout
                epoch_ended = t==self.steps_per_epoch-1

                # End of trajectory/episode handling
                if terminal or epoch_ended:
                    if timeout or epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(obs, dtype=torch.float32).to(self.device))
                    else:
                        v = 0

                    ep_num += 1
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    self.tensorboard_logger.add_scalar('episodic_return_train', ep_ret, epoch*self.steps_per_epoch + (t+1))
                    self.buffer.finish_path(v)
                    obs, ep_ret, ep_len = self.env.reset(), 0, 0
                    # Retrieve training reward
                    x, y = self.logger.load_results(["EpLen", "EpRet"])
                    if len(x) > 0:
                        # Mean training reward over the last 50 episodes
                        mean_reward = np.mean(y[-50:])

                        # New best model
                        if mean_reward > self.best_mean_reward:
                            # print("Num timesteps: {}".format(timestep))
                            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                            self.best_mean_reward = mean_reward
                            self.save_weights(fname=f"best_{trial_num}.pth")
                        
                        if self.env.spec.reward_threshold is not None and self.best_mean_reward >= self.env.spec.reward_threshold:
                            print("Solved Environment, stopping iteration...")
                            return

            # update value function and PPO policy update
            self.update()
            self.logger.dump()
            if self.save_freq > 0 and epoch % self.save_freq == 0:
                self.save_weights(fname=f"latest_{trial_num}.pth")

    def learn(self, timesteps, num_trials=1):
        '''
        Function to learn using PPO.
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
        obs, done, ep_ret, ep_len = self.env.reset(), False, 0, 0
        img = []
        if record:
            img.append(self.env.render('rgb_array'))

        if timesteps is not None:
            for i in range(timesteps):
                # Take stochastic action with policy network
                action, _, _ = self.ac.step(torch.as_tensor(obs, dtype=torch.float32).to(self.device))
                obs, reward, done, _ = self.env.step(action)
                if record:
                    img.append(self.env.render('rgb_array'))
                else:
                    self.env.render()
                ep_ret += reward
                ep_len += 1                
        else:
            while not (done or (ep_len==self.max_ep_len)):
                # Take stochastic action with policy network
                action, _, _ = self.ac.step(torch.as_tensor(obs, dtype=torch.float32).to(self.device))
                obs, reward, done, _ = self.env.step(action)
                if record:
                    img.append(self.env.render('rgb_array'))
                else:
                    self.env.render()
                ep_ret += reward
                ep_len += 1

        self.env.training = True
        if record:
            imageio.mimsave(f'{os.path.join(self.save_dir, "recording.gif")}', [np.array(img) for i, img in enumerate(img) if i%2 == 0], fps=29)

        return ep_ret, ep_len      
