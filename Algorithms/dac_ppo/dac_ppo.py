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
from Algorithms.dac_ppo.buffer import Storage
from Algorithms.utils import to_tensor, to_np, sanitise_state_dict, random_sample,LinearSchedule
from torch.optim import Adam, RMSprop
from Algorithms.dac_ppo.core import OptionGaussianActorCriticNet
from torch.utils.tensorboard import SummaryWriter
from Logger.logger import Logger
from tqdm import tqdm

class DAC_PPO:
    '''
    DAC + PPO
    '''
    def __init__(self, env_fn, save_dir, tensorboard_logdir = None, optimizer_class = Adam, weight_decay=0,
                oc_kwargs=dict(), logger_kwargs=dict(), lr=1e-3, optimization_epochs=5, mini_batch_size=64, ppo_ratio_clip=0.2,
                gamma=0.99, rollout_length=2048, beta_weight=0, entropy_weight=0.01, gradient_clip=5, gae_tau=0.95,
                max_ep_len=2000, save_freq=200, seed=0, **kwargs):
        
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
        self.optimizer_class = optimizer_class
        self.weight_decay = weight_decay
        self.optimizer = optimizer_class(self.network.parameters(), self.lr, weight_decay=self.weight_decay)
        self.gamma = gamma
        self.rollout_length = rollout_length
        self.num_options = oc_kwargs['num_options']
        self.beta_weight = beta_weight
        self.entropy_weight = entropy_weight
        self.gradient_clip = gradient_clip
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq

        self.save_dir = save_dir
        self.logger = Logger(**logger_kwargs)        
        self.tensorboard_logdir = tensorboard_logdir
        # self.tensorboard_logger = SummaryWriter(log_dir=tensorboard_logdir)

        self.is_initial_states = to_tensor(np.ones((1))).byte().to(self.device)
        self.prev_options = to_tensor(np.zeros((1))).long().to(self.device)

        self.best_mean_reward = -np.inf

        self.optimization_epochs = optimization_epochs
        self.mini_batch_size = mini_batch_size
        self.ppo_ratio_clip = ppo_ratio_clip
        self.gae_tau = gae_tau
        self.use_gae = self.gae_tau > 0

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
                        else ConvBody(obs_space, oc_kwargs['conv_layer_sizes'], activation, batchnorm=False)
            state_dim = phi_body.latent_dim
        else:
            state_dim = obs_space[0]
            phi_body = DummyBody(state_dim)

        network_fn = lambda: OptionGaussianActorCriticNet(
            state_dim, act_dim,
            num_options=oc_kwargs['num_options'],
            phi_body=phi_body,
            critic_body=FCBody(state_dim, hidden_units=hidden_units, gate=gate),
            option_body_fn=lambda: FCBody(state_dim, hidden_units=hidden_units, gate=gate),
            device=self.device
        )

        return network_fn
 
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
            'oc': self.network.state_dict()
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
        self.best_mean_reward = -np.inf
        self.network = self.network_fn().to(self.device)
        self.optimizer = self.optimizer_class(self.network.parameters(), self.lr, weight_decay=self.weight_decay)

    def record_online_return(self, ep_ret, timestep, ep_len):
        self.tensorboard_logger.add_scalar('episodic_return_train', ep_ret, timestep)
        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
        self.logger.dump()
        # print(f"episode return: {ep_ret}")

    def compute_pi_hat(self, prediction, prev_option, is_initial_states):
        inter_pi = prediction['inter_pi']
        mask = torch.zeros_like(inter_pi)
        mask[:, prev_option] = 1
        beta = prediction['beta']
        pi_hat = (1 - beta) * mask + beta * inter_pi
        is_initial_states = is_initial_states.view(-1, 1).expand(-1, inter_pi.size(1))
        pi_hat = torch.where(is_initial_states, inter_pi, pi_hat)
        return pi_hat

    def compute_pi_bar(self, options, action, mean, std):
        options = options.unsqueeze(-1).expand(-1, -1, mean.size(-1))
        mean = mean.gather(1, options).squeeze(1)
        std = std.gather(1, options).squeeze(1)
        dist = torch.distributions.Normal(mean, std)
        pi_bar = dist.log_prob(action).sum(-1).exp().unsqueeze(-1)
        return pi_bar

    def compute_log_pi_a(self, options, pi_hat, action, mean, std, mdp):
        if mdp == 'hat':
            return pi_hat.add(1e-5).log().gather(1, options)
        elif mdp == 'bar':
            pi_bar = self.compute_pi_bar(options, action, mean, std)
            return pi_bar.add(1e-5).log()
        else:
            raise NotImplementedError

    def compute_adv(self, storage, mdp):
        v = storage.__getattribute__('v_%s' % (mdp))
        adv = storage.__getattribute__('adv_%s' % (mdp))
        all_ret = storage.__getattribute__('ret_%s' % (mdp))

        ret = v[-1].detach()
        advantages = to_tensor(np.zeros((1))).to(self.device)
        for i in reversed(range(self.rollout_length)):
            ret = storage.r[i] + self.gamma * storage.m[i] * ret
            if not self.use_gae:
                advantages = ret - v[i].detach()
            else:
                td_error = storage.r[i] + self.gamma * storage.m[i] * v[i + 1] - v[i]
                advantages = advantages * self.gae_tau * self.gamma * storage.m[i] + td_error
            adv[i] = advantages.detach()
            all_ret[i] = ret.detach()

    def update(self, storage, mdp, timestep, freeze_v=False):
        states, actions, options, log_probs_old, returns, advantages, prev_options, inits, pi_hat, mean, std = \
            storage.cat(
                ['s', 'a', 'o', 'log_pi_%s' % (mdp), 'ret_%s' % (mdp), 'adv_%s' % (mdp), 'prev_o', 'init', 'pi_hat',
                 'mean', 'std'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        pi_hat = pi_hat.detach()
        mean = mean.detach()
        std = std.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        for _ in range(self.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), self.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = to_tensor(batch_indices).long()

                sampled_pi_hat = pi_hat[batch_indices]
                sampled_mean = mean[batch_indices]
                sampled_std = std[batch_indices]
                sampled_states = states[batch_indices]
                sampled_prev_o = prev_options[batch_indices]
                sampled_init = inits[batch_indices]

                sampled_options = options[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = self.network(sampled_states, unsqueeze=False)
                if mdp == 'hat':
                    cur_pi_hat = self.compute_pi_hat(prediction, sampled_prev_o.view(-1), sampled_init.view(-1))
                    entropy = -(cur_pi_hat * cur_pi_hat.add(1e-5).log()).sum(-1).mean()
                    log_pi_a = self.compute_log_pi_a(
                        sampled_options, cur_pi_hat, sampled_actions, sampled_mean, sampled_std, mdp)
                    beta_loss = prediction['beta'].mean()
                elif mdp == 'bar':
                    log_pi_a = self.compute_log_pi_a(
                        sampled_options, sampled_pi_hat, sampled_actions, prediction['mean'], prediction['std'], mdp)
                    entropy = 0
                    beta_loss = 0
                else:
                    raise NotImplementedError

                if mdp == 'bar':
                    v = prediction['q_o'].gather(1, sampled_options)
                elif mdp == 'hat':
                    v = (prediction['q_o'] * sampled_pi_hat).sum(-1).unsqueeze(-1)
                else:
                    raise NotImplementedError

                ratio = (log_pi_a - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.ppo_ratio_clip,
                                          1.0 + self.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - self.entropy_weight * entropy + \
                              self.beta_weight * beta_loss

                # discarded = (obj > obj_clipped).float().mean()
                value_loss = 0.5 * (sampled_returns - v).pow(2).mean()

                self.tensorboard_logger.add_scalar(f"loss/{mdp}_value_loss", value_loss.item(), timestep)
                self.tensorboard_logger.add_scalar(f"loss/{mdp}_policy_loss", policy_loss.item(), timestep)
                self.tensorboard_logger.add_scalar(f"loss/{mdp}_beta_loss", beta_loss if isinstance(beta_loss, int) else beta_loss.item(), timestep)
                
                if freeze_v:
                    value_loss = 0

                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
                self.optimizer.step()

    def learn_one_trial(self, num_timesteps, trial_num=1):
        self.states, ep_ret, ep_len = self.env.reset(), 0, 0
        storage = Storage(self.rollout_length, ['adv_bar', 'adv_hat', 'ret_bar', 'ret_hat'])
        states = self.states
        for timestep in tqdm(range(1, num_timesteps+1)):
            prediction = self.network(states)
            pi_hat = self.compute_pi_hat(prediction, self.prev_options, self.is_initial_states)
            dist = torch.distributions.Categorical(probs=pi_hat)
            options = dist.sample()

            # Gaussian policy
            mean = prediction['mean'][0, options]
            std = prediction['std'][0, options]
            dist = torch.distributions.Normal(mean, std)

            # select action
            actions = dist.sample()

            pi_bar = self.compute_pi_bar(options.unsqueeze(-1), actions,
                                         prediction['mean'], prediction['std'])

            v_bar = prediction['q_o'].gather(1, options.unsqueeze(-1))
            v_hat = (prediction['q_o'] * pi_hat).sum(-1).unsqueeze(-1)

            next_states, rewards, terminals, _ = self.env.step(to_np(actions[0]))
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
                         'a': actions,
                         'o': options.unsqueeze(-1),
                         'prev_o': self.prev_options.unsqueeze(-1),
                         's': to_tensor(states).unsqueeze(0),
                         'init': self.is_initial_states.unsqueeze(-1),
                         'pi_hat': pi_hat,
                         'log_pi_hat': pi_hat[0, options].add(1e-5).log().unsqueeze(-1),
                         'log_pi_bar': pi_bar.add(1e-5).log(),
                         'v_bar': v_bar,
                         'v_hat': v_hat})

            self.is_initial_states = to_tensor(terminals).unsqueeze(-1).to(self.device).byte()
            self.prev_options = options
            states = next_states
           
            if timestep%self.rollout_length==0:
                self.states = states
                prediction = self.network(states)
                pi_hat = self.compute_pi_hat(prediction, self.prev_options, self.is_initial_states)
                dist = torch.distributions.Categorical(pi_hat)
                options = dist.sample()
                v_bar = prediction['q_o'].gather(1, options.unsqueeze(-1))
                v_hat = (prediction['q_o'] * pi_hat).sum(-1).unsqueeze(-1)

                storage.add(prediction)
                storage.add({
                    'v_bar': v_bar,
                    'v_hat': v_hat,
                })
                storage.placeholder()

                self.compute_adv(storage, 'bar')
                self.compute_adv(storage, 'hat')
                mdps = ['hat', 'bar']
                np.random.shuffle(mdps)
                self.update(storage, mdps[0], timestep)
                self.update(storage, mdps[1], timestep)
                
                storage = Storage(self.rollout_length, ['adv_bar', 'adv_hat', 'ret_bar', 'ret_hat'])
            
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
        self.network.eval()
        if render:
            self.env.render('human')
        states, terminals, ep_ret, ep_len = self.env.reset(), False, 0, 0
        is_initial_states = to_tensor(np.ones((1))).byte().to(self.device)
        prev_options = to_tensor(np.zeros((1))).long().to(self.device)
        img = []
        if record:
            img.append(self.env.render('rgb_array'))

        if timesteps is not None:
            for i in range(timesteps):
                prediction = self.network(states)
                pi_hat = self.compute_pi_hat(prediction, prev_options, is_initial_states)
                dist = torch.distributions.Categorical(probs=pi_hat)
                options = dist.sample()

                # Gaussian policy
                mean = prediction['mean'][0, options]
                std = prediction['std'][0, options]
                dist = torch.distributions.Normal(mean, std)

                # select action
                actions = mean
                
                next_states, rewards, terminals, _ = self.env.step(to_np(actions[0]))
                is_initial_states = to_tensor(terminals).unsqueeze(-1).byte().to(self.device)
                prev_options = options
                states = next_states
                if record:
                    img.append(self.env.render('rgb_array'))
                else:
                    self.env.render()
                ep_ret += rewards
                ep_len += 1                
        else:
            while not (terminals or (ep_len==self.max_ep_len)):
                # select option
                prediction = self.network(states)
                pi_hat = self.compute_pi_hat(prediction, prev_options, is_initial_states)
                dist = torch.distributions.Categorical(probs=pi_hat)
                options = dist.sample()

                # Gaussian policy
                mean = prediction['mean'][0, options]
                std = prediction['std'][0, options]
                # dist = torch.distributions.Normal(mean, std)

                # select action
                actions = mean

                next_states, rewards, terminals, _ = self.env.step(to_np(actions[0]))
                is_initial_states = to_tensor(terminals).unsqueeze(-1).byte().to(self.device)
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
