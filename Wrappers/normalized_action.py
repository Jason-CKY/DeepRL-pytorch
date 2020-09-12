import gym


class NormalizedActions(gym.ActionWrapper):
    '''
    Assuming actor model outputs tanh range [-1, 1]
    Convert the range [-1, 1] to [env.action_space.low, env.action_space.high]
    '''
    def action(self, action):
        """
        Normalizes the actions to be in between action_space.high and action_space.low.
        If action_space.low == -action_space.high, this is equals to action_space.high*action.
        :param action:
        :return: normalized action
        """
        action = (action + 1) / 2  # [-1, 1] => [0, 1] 
        action *= (self.action_space.high - self.action_space.low) # [0, 1] => [0, high-low] adjust range of outputs
        action += self.action_space.low # [low, high]
        return action

    def reverse_action(self, action):
        """
        Reverts the normalization
        :param action:
        :return:
        """
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action