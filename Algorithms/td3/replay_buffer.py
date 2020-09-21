import numpy as np
from collections import deque
import random
import torch
import pickle

class ReplayBuffer:
    '''
    A FIFO experience replay buffer to store 
    '''
    def __init__(self, size):
        """
        Args:
            size (integer): The size of the replay buffer.
        """
        size = int(size)
        self.buffer = deque(maxlen=size)
        self.max_size = size

    def append(self, state, action, reward, next_state, terminal):
        '''
        Args:
            state (Numpy array): The state.              
            action (integer): The action.s
            reward (float): The reward.
            terminal (integer): 1 if the next state is a terminal state and 0 otherwise.
            next_state (Numpy array): The next state. 
        '''
        self.buffer.append([state, action, reward, next_state, terminal])

    def sample(self, batch_size):
        '''
        Returns:
            A list of transition tuples including state, action, reward, next state and terminal
        '''
        sample = random.sample(self.buffer, batch_size)
        states = []
        actions = []
        rewards = []
        terminals = []
        next_states = []
        for experience in sample:
            state, action, reward, next_state, terminal = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            terminals.append(terminal)
            next_states.append(next_state)
        
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.float32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        terminals = torch.as_tensor(terminals, dtype=torch.float32)
        return states, actions, rewards, next_states, terminals

    def size(self):
        return len(self.buffer)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.buffer = pickle.load(f)
        assert self.buffer.maxlen == self.max_size, "Attempted to load buffer with different max size"
