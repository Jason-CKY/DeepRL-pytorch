import gym
import pybullet_envs
import torch
import numpy as np
import time
import argparse
import os
import imageio

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from Algorithms.trpo.core import MLPActorCritic, CNNActorCritic
from Algorithms.utils import get_actor_critic_module, sanitise_state_dict
from Algorithms.trpo.gae_buffer import GAEBuffer
from Logger.logger import Logger
from copy import deepcopy
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class TRPO:
    
    def __init__(self, env_fn, save_dir, ac_kwargs=dict(), seed=0, tensorboard_logdir = None,
         steps_per_epoch=400, batch_size=400, gamma=0.99, delta=0.01, vf_lr=1e-3,
         train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10, 
         backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, logger_kwargs=dict(), 
         save_freq=10, algo='trpo', ngpu=1):
        """
        Trust Region Policy Optimization 
        (with support for Natural Policy Gradient)
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
            delta (float): KL-divergence limit for TRPO / NPG update. 
                (Should be small for stability. Values like 0.01, 0.05.)
            vf_lr (float): Learning rate for value function optimizer.
            train_v_iters (int): Number of gradient descent steps to take on 
                value function per epoch.
            damping_coeff (float): Artifact for numerical stability, should be 
                smallish. Adjusts Hessian-vector product calculation:
                
                .. math:: Hv \\rightarrow (\\alpha I + H)v
                where :math:`\\alpha` is the damping coefficient. 
                Probably don't play with this hyperparameter.
            cg_iters (int): Number of iterations of conjugate gradient to perform. 
                Increasing this will lead to a more accurate approximation
                to :math:`H^{-1} g`, and possibly slightly-improved performance,
                but at the cost of slowing things down. 
                Also probably don't play with this hyperparameter.
            backtrack_iters (int): Maximum number of steps allowed in the 
                backtracking line search. Since the line search usually doesn't 
                backtrack, and usually only steps back once when it does, this
                hyperparameter doesn't often matter.
            backtrack_coeff (float): How far back to step during backtracking line
                search. (Always between 0 and 1, usually above 0.5.)
            lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
                close to 1.)
            max_ep_len (int): Maximum length of trajectory / episode / rollout.
            logger_kwargs (dict): Keyword args for Logger. 
                            (1) output_dir = None
                            (2) output_fname = 'progress.pickle'
            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.
            algo: Either 'trpo' or 'npg': this code supports both, since they are 
                almost the same.
        """
        # logger stuff
        self.logger = Logger(**logger_kwargs)

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = env_fn()
        self.vf_lr = vf_lr
        self.steps_per_epoch = steps_per_epoch # if steps_per_epoch > self.env.spec.max_episode_steps else self.env.spec.max_episode_steps        
        self.max_ep_len = max_ep_len
        self.train_v_iters = train_v_iters

        # Main network
        self.ngpu = ngpu
        self.actor_critic = get_actor_critic_module(ac_kwargs, 'trpo')
        self.ac_kwargs = ac_kwargs
        self.ac = self.actor_critic(self.env.observation_space, self.env.action_space, device=self.device, ngpu=self.ngpu, **ac_kwargs)

        # Create Optimizers
        self.v_optimizer = optim.Adam(self.ac.v.parameters(), lr=self.vf_lr)

        # GAE buffer
        self.gamma = gamma
        self.lam = lam
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        self.buffer = GAEBuffer(self.obs_dim, self.act_dim, self.steps_per_epoch, self.device, self.gamma, self.lam)
        self.batch_size = batch_size

        self.cg_iters = cg_iters
        self.damping_coeff = damping_coeff
        self.delta = delta
        self.backtrack_coeff = backtrack_coeff
        self.algo = algo
        self.backtrack_iters = backtrack_iters
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
        self.buffer = GAEBuffer(self.obs_dim, self.act_dim, self.steps_per_epoch, self.device, self.gamma, self.lam)

    def flat_grad(self, grads, hessian=False):
        grad_flatten = []
        if hessian == False:
            for grad in grads:
                grad_flatten.append(grad.view(-1))
            grad_flatten = torch.cat(grad_flatten)
            return grad_flatten
        elif hessian == True:
            for grad in grads:
                grad_flatten.append(grad.contiguous().view(-1))
            grad_flatten = torch.cat(grad_flatten).data
            return grad_flatten

    def cg(self, obs, b, EPS=1e-8, residual_tol=1e-10):
        # Conjugate gradient algorithm
        # (https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        x = torch.zeros(b.size()).to(self.device)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r).to(self.device)

        for _ in range(self.cg_iters):
            Ap = self.hessian_vector_product(obs, p)
            alpha = rdotr / (torch.dot(p, Ap).to(self.device) + EPS)
            
            x += alpha * p
            r -= alpha * Ap
            
            new_rdotr = torch.dot(r, r)
            p = r + (new_rdotr / rdotr) * p
            rdotr = new_rdotr

            if rdotr < residual_tol:
                break

        return x

    def hessian_vector_product(self, obs, p):
        p = p.detach()
        kl = self.ac.pi.calculate_kl(old_policy=self.ac.pi, new_policy=self.ac.pi, obs=obs)
        kl_grad = torch.autograd.grad(kl, self.ac.pi.parameters(), create_graph=True)
        kl_grad = self.flat_grad(kl_grad)

        kl_grad_p = (kl_grad * p).sum() 
        kl_hessian = torch.autograd.grad(kl_grad_p, self.ac.pi.parameters())
        kl_hessian = self.flat_grad(kl_hessian, hessian=True)
        return kl_hessian + p * self.damping_coeff

    def flat_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        params_flatten = torch.cat(params)
        return params_flatten

    def update_model(self, model, new_params):
        index = 0
        for params in model.parameters():
            params_length = len(params.view(-1))
            new_param = new_params[index: index + params_length]
            new_param = new_param.view(params.size())
            params.data.copy_(new_param)
            index += params_length

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

            # Prediction logπ_old(s), logπ(s)
            _, logp = self.ac.pi(obs, act)
            
            # Policy loss
            ratio_old = torch.exp(logp - logp_old)
            surrogate_adv_old = (ratio_old*adv).mean()
            
            # policy gradient calculation as per algorithm, flatten to do matrix calculations later
            gradient = torch.autograd.grad(surrogate_adv_old, self.ac.pi.parameters()) # calculate gradient of policy loss w.r.t to policy parameters
            gradient = self.flat_grad(gradient)

            # Core calculations for NPG/TRPO
            search_dir = self.cg(obs, gradient.data)    # H^-1 g
            gHg = (self.hessian_vector_product(obs, search_dir) * search_dir).sum(0)
            step_size = torch.sqrt(2 * self.delta / gHg)
            old_params = self.flat_params(self.ac.pi)
            # update the old model, calculate KL divergence then decide whether to update new model
            self.update_model(self.ac.pi_old, old_params)        

            if self.algo == 'npg':
                params = old_params + step_size * search_dir
                self.update_model(self.ac.pi, params)

                kl = self.ac.pi.calculate_kl(new_policy=self.ac.pi, old_policy=self.ac.pi_old, obs=obs)
            elif self.algo == 'trpo':
                for i in range(self.backtrack_iters):
                    # Backtracking line search
                    # (https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) 464p.
                    params = old_params + (self.backtrack_coeff**(i+1)) * step_size * search_dir
                    self.update_model(self.ac.pi, params)

                    # Prediction logπ_old(s), logπ(s)
                    _, logp = self.ac.pi(obs, act)
                    
                    # Policy loss
                    ratio = torch.exp(logp - logp_old)
                    surrogate_adv = (ratio*adv).mean()

                    improve = surrogate_adv - surrogate_adv_old
                    kl = self.ac.pi.calculate_kl(new_policy=self.ac.pi, old_policy=self.ac.pi_old, obs=obs)
                    
                    # print(f"kl: {kl}")
                    if kl <= self.delta and improve>0:
                        print('Accepting new params at step %d of line search.'%i)
                        # self.backtrack_iters.append(i)
                        # log backtrack_iters=i
                        break

                    if i == self.backtrack_iters-1:
                        print('Line search failed! Keeping old params.')
                        # self.backtrack_iters.append(i)
                        # log backtrack_iters=i

                        params = self.flat_params(self.ac.pi_old)
                        self.update_model(self.ac.pi, params)

            # Update Critic
            for _ in range(self.train_v_iters):
                self.v_optimizer.zero_grad()
                v = self.ac.v(obs)
                v_loss = ((v-ret)**2).mean()
                v_loss.backward()
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
            'v_optimizer': self.v_optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(self.save_dir, _fname))
        self.env.save(os.path.join(self.save_dir, "env.json"))
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

            # update value function and TRPO policy update
            self.update()
            self.logger.dump()
            if self.save_freq > 0 and epoch % self.save_freq == 0:
                self.save_weights(fname=f"latest_{trial_num}.pth")
            
    def learn(self, timesteps, num_trials=1):
        '''
        Function to learn using TRPO.
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
