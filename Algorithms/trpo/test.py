from trpo import TRPO
import gym
import pybullet_envs

# x = TRPO(lambda: gym.make('CartPoleContinuousBulletEnv-v0'))
x = TRPO(lambda: gym.make('CartPole-v0'))
x.learn(1000000)