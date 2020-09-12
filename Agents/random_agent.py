from .base_agent import BaseAgent
import os

class Random_Agent(BaseAgent):
    def __init__(self):
        pass
    
    def agent_init(self, agent_config):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the agent.

        Assume agent_config dict contains:
        {
            action_space: environment action space
        }
        """
        self.action_space = agent_config['action_space']
        self.last_state = None
        self.last_action = None

        self.sum_rewards = 0
        self.episode_steps = 0

    def optimize_network(self, experiences):
        """
        Args:
            experiences (Numpy array): The batch of experiences including the states, actions, 
                                    rewards, terminals, and next_states.
        """
        pass

    def policy(self, state):
        """
        Args:
            state (Numpy array)/(torch tensor)/(list): the state
        Returns:
            the action
        if state is in the shape [n, state_dim], output will be [n, action_dim]
        if state is in the shape [state_dim], output will be [action_dim]
        """
        return self.action_space.sample()

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """
        self.last_action = self.policy(self.last_state)
        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        self.sum_rewards += reward
        self.episode_steps += 1

        action = self.policy(state)
                
        # Update the last state and last action.
        self.last_state = state
        self.last_action = action
        
        return action
        
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1


    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        pass

    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")

    def set_train(self):
        '''
        Set actor and critic networks into train mode
        '''
        pass

    def set_eval(self):
        '''
        Set actor and critic networks into eval mode
        '''
        pass
    
    def save_checkpoint(self, episode_num, solved=False):
        pass
    
    def get_latest_path(self):
        """
        get the latest created file in the checkpoint directory
        Returns:
            the latest saved model weights
        """
        pass

    def load_checkpoint(self, checkpoint_path=None):
        """
        load networks and optimizer paramters from checkpoint_path
        if checkpoint_path is None, use the latest created path from checkpoint_dir
        Args:
            checkpoint_path: path to checkpoint
        """
        pass 

