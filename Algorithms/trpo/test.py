from trpo import TRPO
import gym
import pybullet_envs

logger_kwargs = {
    "output_dir": save_dir
}
# x = TRPO(lambda: gym.make('CartPoleContinuousBulletEnv-v0'))
x = TRPO(lambda: gym.make('CartPoleBulletEnv-v1'), logger_kwargs=logger_kwargs)
x.learn(1000000)