#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import os
import torch
import torch.nn as nn
import numpy as np
import imageio
from torch.nn import functional as F
from Algorithms.body import VAE, ConvBody, FCBody, DummyBody
from Algorithms.option_critic.buffer import Storage
from Algorithms.utils import to_tensor, to_np, sanitise_state_dict,LinearSchedule
from torch.optim import Adam, RMSprop
from Algorithms.option_critic.core import OptionCriticNet
from torch.utils.tensorboard import SummaryWriter
from Logger.logger import Logger
from tqdm import tqdm

class Option_Critic:
    def __init__(self, env_fn, save_dir, tensorboard_logdir = None, optimizer_class = RMSprop,
                oc_kwargs=dict(), logger_kwargs=dict(), eps_start=1.0, eps_end=0.1, eps_decay=1e4, lr=1e-3, 
                gamma=0.99, rollout_length=2048, beta_reg=0.01, entropy_weight=0.01, gradient_clip=5,
                target_network_update_freq=200, max_ep_len=2000, save_freq=200, seed=0, **kwargs):
        
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = lr
        self.env_fn = env_fn
        self.env = env_fn()
        self.oc_kwargs = oc_kwargs
        self.network_fn = self.get_network_fn(self.oc_kwargs)
        self.network = self.network_fn().to(self.device)
        self.target_network = self.network_fn().to(self.device)
        self.optimizer_class = optimizer_class
        self.optimizer = optimizer_class(self.network.parameters(), self.lr)
        self.target_network.load_state_dict(self.network.state_dict())
        self.eps_start = eps_start; self.eps_end = eps_end; self.eps_decay = eps_decay
        self.eps_schedule = LinearSchedule(eps_start, eps_end, eps_decay)
        self.gamma = gamma
        self.rollout_length = rollout_length
        self.num_options = oc_kwargs['num_options']
        self.beta_reg = beta_reg
        self.entropy_weight = entropy_weight
        self.gradient_clip = gradient_clip
        self.target_network_update_freq = target_network_update_freq
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq

        self.save_dir = save_dir
        self.logger = Logger(**logger_kwargs)        
        self.tensorboard_logdir = tensorboard_logdir
        # self.tensorboard_logger = SummaryWriter(log_dir=tensorboard_logdir)

        self.is_initial_states = to_tensor(np.ones((1))).byte()
        self.prev_options = self.is_initial_states.clone().long().to(self.device)

        self.best_mean_reward = -np.inf

    def get_network_fn(self, oc_kwargs):
        activation=nn.ReLU
        gate = F.relu
        obs_space = self.env.observation_space.shape
        hidden_units = oc_kwargs['hidden_sizes']
        act_dim = self.env.action_space.shape[0]
        self.continuous = True

        if len(obs_space) > 1:
            # image observations
            phi_body = VAE(load_path =oc_kwargs['vae_weights_path'], device=self.device) if oc_kwargs['model_type'].lower() == 'vae' \
                        else ConvBody(obs_space, oc_kwargs['conv_layer_sizes'], activation, batchnorm=True)
            state_dim = phi_body.latent_dim
        else:
            state_dim = obs_space[0]
            phi_body = FCBody(state_dim, hidden_units=hidden_units, gate=gate)

        network_fn = lambda: OptionCriticNet(
            body=phi_body,
            action_dim=act_dim,
            num_options=oc_kwargs['num_options'],
            device=self.device
        )

        return network_fn
        
    def update(self, storage, states, timestep):
        with torch.no_grad():
            prediction = self.target_network(states)
            storage.placeholder() # create the beta_adv attribute inside storage to be [None]*rollout_length
            betas = prediction['beta'].squeeze()[self.prev_options]
            ret = (1 - betas) * prediction['q'][self.worker_index, self.prev_options] + \
                  betas * torch.max(prediction['q'], dim=-1)[0]
            ret = ret.unsqueeze(-1)

        for i in reversed(range(self.rollout_length)):
            ret = storage.r[i] + self.gamma * storage.m[i] * ret
            adv = ret - storage.q[i].gather(1, storage.o[i])
            storage.ret[i] = ret
            storage.adv[i] = adv

            v = storage.q[i].max(dim=-1, keepdim=True)[0] * (1 - storage.eps[i]) + storage.q[i].mean(-1).unsqueeze(-1) * storage.eps[i]
            q = storage.q[i].gather(1, storage.prev_o[i])
            storage.beta_adv[i] = q - v + self.beta_reg
        
        q, beta, log_pi, ret, adv, beta_adv, ent, option, action, initial_states, prev_o = \
            storage.cat(['q', 'beta', 'log_pi', 'ret', 'adv', 'beta_adv', 'ent', 'o', 'a', 'init', 'prev_o'])

        # calculate loss function
        q_loss = (q.gather(1, option) - ret.detach()).pow(2).mul(0.5).mean()
        pi_loss = -(log_pi.gather(1, action) * adv.detach()) - self.entropy_weight * ent
        pi_loss = pi_loss.mean()
        beta_loss = (beta.gather(1, prev_o) * beta_adv.detach() * (1 - initial_states)).mean()
        # logging all losses
        self.logger.store(q_loss=q_loss.item(), pi_loss=pi_loss.item(), beta_loss=beta_loss.item())
        self.tensorboard_logger.add_scalar("loss/q_loss", q_loss.item(), timestep)
        self.tensorboard_logger.add_scalar("loss/pi_loss", pi_loss.item(), timestep)
        self.tensorboard_logger.add_scalar("loss/beta_loss", beta_loss.item(), timestep)

        # backward and train
        self.optimizer.zero_grad()
        (pi_loss + q_loss + beta_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.optimizer.step()

    def save_weights(self, best=False, fname=None):
        '''
        save the pytorch model weights of ac and ac_targ
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
            'oc': self.network.state_dict(),
            'oc_target': self.target_network.state_dict()
        }
        torch.save(checkpoint, os.path.join(self.save_dir, _fname))
        self.env.save(os.path.join(self.save_dir, "env.json"))
        print(f"checkpoint saved at {os.path.join(self.save_dir, _fname)}")

    def load_weights(self, best=True, fname=None):
        '''
        Load the model weights and replay buffer from self.save_dir
        Args:
            best (bool): If True, save from the weights file with the best mean episode reward
            load_buffer (bool): If True, load the replay buffer from the pickled file
        '''
        if fname is not None:
            _fname = fname
        elif best:
            _fname = "best.pth"
        else:
            _fname = "model_weights.pth"
        checkpoint_path = os.path.join(self.save_dir, _fname)
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.network.load_state_dict(sanitise_state_dict(checkpoint['oc']))
            self.target_network.load_state_dict(sanitise_state_dict(checkpoint['oc_target']))

            env_path = os.path.join(self.save_dir, "env.json")
            if os.path.isfile(env_path):
                self.env = self.env.load(env_path)
                print("Environment loaded")
            
            print('checkpoint loaded at {}'.format(checkpoint_path))
        else:
            raise OSError("Checkpoint file not found.")    

    def reinit_network(self):
        self.seed += 1
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.network = self.network_fn().to(self.device)
        self.target_network = self.network_fn().to(self.device)
        self.optimizer = self.optimizer_class(self.network.parameters(), self.lr)
        self.target_network.load_state_dict(self.network.state_dict())
        self.eps_schedule = LinearSchedule(self.eps_start, self.eps_end, self.eps_decay)

    def sample_option(self, prediction, epsilon, prev_option, is_intial_states):
        with torch.no_grad():
            # get q value
            q_option = prediction['q_o']
            pi_option = torch.zeros_like(q_option).add(epsilon / q_option.size(1))

            # greedy policy
            greedy_option = q_option.argmax(dim=-1, keepdim=True)
            prob = 1 - epsilon + epsilon / q_option.size(1)
            prob = torch.zeros_like(pi_option).add(prob)
            pi_option.scatter_(1, greedy_option, prob)

            mask = torch.zeros_like(q_option)
            mask[:, prev_option] = 1
            beta = prediction['beta']
            pi_hat_option = (1 - beta) * mask + beta * pi_option

            dist = torch.distributions.Categorical(probs=pi_option)
            options = dist.sample()
            dist = torch.distributions.Categorical(probs=pi_hat_option)
            options_hat = dist.sample()

            options = torch.where(is_intial_states.to(self.device), options, options_hat)
        return options

    def record_online_return(self, ep_ret, timestep, ep_len):
        self.tensorboard_logger.add_scalar('episodic_return_train', ep_ret, timestep)
        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
        self.logger.dump()
        # print(f"episode return: {ep_ret}")

    def learn_one_trial(self, num_timesteps, trial_num=1):
        self.states, ep_ret, ep_len = self.env.reset(), 0, 0
        storage = Storage(self.rollout_length, ['beta', 'o', 'beta_adv', 'prev_o', 'init', 'eps'])
        for timestep in tqdm(range(1, num_timesteps+1)):
            prediction = self.network(self.states)
            epsilon = self.eps_schedule()
            # select option
            options = self.sample_option(prediction, epsilon, self.prev_options, self.is_initial_states)
            prediction['pi'] = prediction['pi'][0, options]
            prediction['log_pi'] = prediction['log_pi'][0, options]
            dist = torch.distributions.Categorical(probs=prediction['pi'])
            actions = dist.sample()
            entropy = dist.entropy()

            next_states, rewards, terminals, _ = self.env.step(to_np(actions))
            ep_ret += rewards
            ep_len += 1
            
            # end of episode handling
            if terminals or ep_len == self.max_ep_len:
                next_states = self.env.reset()
                self.record_online_return(ep_ret, timestep, ep_len)
                ep_ret, ep_len = 0, 0
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

            storage.add(prediction)
            storage.add({'r': to_tensor(rewards).to(self.device).unsqueeze(-1),
                        'm': to_tensor(1 - terminals).to(self.device).unsqueeze(-1),
                        'o': options.unsqueeze(-1),
                        'prev_o': self.prev_options.unsqueeze(-1),
                        'ent': entropy,
                        'a': actions.unsqueeze(-1),
                        'init': self.is_initial_states.unsqueeze(-1).to(self.device).float(),
                        'eps': epsilon})

            self.is_initial_states = to_tensor(terminals).unsqueeze(-1).byte()
            self.prev_options = options
            self.states = next_states

            if timestep % self.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
           
            if timestep%self.rollout_length==0:
                self.update(storage, self.states, timestep)
                storage = Storage(self.rollout_length, ['beta', 'o', 'beta_adv', 'prev_o', 'init', 'eps'])
            
            if self.save_freq > 0 and timestep % self.save_freq == 0:
                self.save_weights(fname=f"latest_{trial_num}.pth")

    def learn(self, timesteps, num_trials=1):
        '''
        Function to learn using DDPG.
        Args:
            timesteps (int): number of timesteps to train for
        '''
        self.env.training=True
        self.network.train()
        self.target_network.train()
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
        self.env.training=False
        self.network.eval(); self.target_network.eval()
        if render:
            self.env.render('human')
        states, done, ep_ret, ep_len = self.env.reset(), False, 0, 0
        is_initial_states = to_tensor(np.ones((1))).byte().to(self.device)
        prev_options = is_initial_states.clone().long().to(self.device)
        prediction = self.network(states)
        epsilon = 0.0
        # select option
        options = self.sample_option(prediction, epsilon, prev_options, is_initial_states)
        img = []
        if record:
            img.append(self.env.render('rgb_array'))

        if timesteps is not None:
            for i in range(timesteps):
                # select option
                options = self.sample_option(prediction, epsilon, prev_options, is_initial_states)

                # Gaussian policy
                mean = prediction['mean'][0, options]
                std = prediction['std'][0, options]
                dist = torch.distributions.Normal(mean, std)

                # select action
                actions = dist.sample()

                next_states, rewards, terminals, _ = self.env.step(to_np(actions[0]))
                is_initial_states = to_tensor(terminals).unsqueeze(-1).byte()
                prev_options = options
                states = next_states
                if record:
                    img.append(self.env.render('rgb_array'))
                else:
                    self.env.render()
                ep_ret += rewards
                ep_len += 1                
        else:
            while not (done or (ep_len==self.max_ep_len)):
                # select option
                options = self.sample_option(prediction, epsilon, prev_options, is_initial_states)

                # Gaussian policy
                mean = prediction['mean'][0, options]
                std = prediction['std'][0, options]
                dist = torch.distributions.Normal(mean, std)
                # select action
                actions = dist.sample()

                next_states, rewards, terminals, _ = self.env.step(to_np(actions[0]))
                is_initial_states = to_tensor(terminals).unsqueeze(-1).byte()
                prev_options = options
                states = next_states
                if record:
                    img.append(self.env.render('rgb_array'))
                else:
                    self.env.render()


                ep_ret += rewards
                ep_len += 1

        if record:
            imageio.mimsave(f'{os.path.join(self.save_dir, "recording.gif")}', [np.array(img) for i, img in enumerate(img) if i%2 == 0], fps=29)

        self.env.training=True
        return ep_ret, ep_len      
