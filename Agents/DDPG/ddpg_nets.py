import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def fanin_init(size):
    fanin = size[0] # weights.data.size() gives [out_features, in_features]
    v = 1./np.sqrt(fanin)
    return torch.FloatTensor(size).uniform_(-v, v)

class Critic(nn.Module):
    """ Critic Model Architecture for Agent
    """ 
    def __init__(self, critic_config, h1=400, h2=300):
        '''
        Assume critic_config:dictionary contains:
            state_dim: int
            action_dim: int
        '''
        super(Critic, self).__init__()
        state_dim = critic_config['state_dim']
        action_dim = critic_config['action_dim']
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc1.bias.data = fanin_init(self.fc1.bias.data.size())

        self.fc2 = nn.Linear(h1+action_dim, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc2.bias.data = fanin_init(self.fc2.bias.data.size())

        self.fc3 = nn.Linear(h2, 1)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        '''
        Args:
            states: pytorch tensor of shape [n, state_dim]
            actions: pytorch tensor of shape [n, action_dim]
        '''
        s1 = F.relu(self.fc1(states))
        x = torch.cat([s1, actions], dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Actor(nn.Module):
    """ Actor Model Architecture for Agent
    """ 

    def __init__(self, actor_config, h1=400, h2=300):
        '''
        Assume actor_config:dictionary contains:
            state_dim: int
            action_dim: int
        '''
        super(Actor, self).__init__()
        state_dim = actor_config['state_dim']
        action_dim = actor_config['action_dim']
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc1.bias.data = fanin_init(self.fc1.bias.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc2.bias.data = fanin_init(self.fc2.bias.data.size())

        self.fc3 = nn.Linear(h2, action_dim)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        '''
        Args:
            states: pytorch tensor of shape [n, state_dim]
        '''
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.torch.tanh(self.fc3(x))
        return x