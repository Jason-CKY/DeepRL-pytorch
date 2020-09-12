import numpy as np
from collections import namedtuple, deque
import random

class ReplayBuffer:
    """Replay Buffer class for the agent
    Note:
        Stores a list of current state, action, reward, is_terminal, and next state 
        sampled from the environment.
        Able to output a random state, action, reward, next state from the buffer
    """ 
    def __init__(self, size, minibatch_size, seed):
        """
        Args:
            size (integer): The size of the replay buffer.              
            minibatch_size (integer): The sample size.
            seed (integer): The seed for the random number generator. 
        """
        self.buffer = deque(maxlen=size)
        self.minibatch_size = minibatch_size 
        self.seed = random.seed(seed)
        # self.rand_generator = np.random.RandomState(seed)
        self.max_size = size

    def append(self, state, action, reward, terminal, next_state):
        '''
        Args:
            state (Numpy array): The state.              
            action (integer): The action.
            reward (float): The reward.
            terminal (integer): 1 if the next state is a terminal state and 0 otherwise.
            next_state (Numpy array): The next state. 
        '''
        self.buffer.append([state, action, reward, terminal, next_state])

    def sample(self):
        '''
        Returns:
            A list of transition tuples including state, action, reward, terminal, and next state
        '''
        # idxs = self.rand_generator.choice(np.arange(len(self.buffer)), size=self.minibatch_size)
        # sample = [self.buffer[idx] for idx in idxs]
        sample = random.sample(self.buffer, self.minibatch_size)
        states = []
        actions = []
        rewards = []
        terminals = []
        next_states = []
        for experience in sample:
            state, action, reward, terminal, next_state = experience
            states.append(state)
            actions.append(action)
            rewards.append([reward])
            terminals.append([terminal])
            next_states.append(next_state)
        
        return states, actions, rewards, terminals, next_states

    def size(self):
        return len(self.buffer)