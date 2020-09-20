import os
import numpy as np
import matplotlib.pyplot as plt
from Logger.logger import Logger
import argparse

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def cumulative_sum(x):
    x = np.array(x)
    output = []
    for i in range(len(x)):
        output.append(x[:i+1].sum())
    return output

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='Model_Weights\CartPoleContinuousBulletEnv-v0\ddpg', help='environment_id')
    parser.add_argument('--save', action='store_true', help='if true, save the plot to log directory')
  
    return parser.parse_args()

def main():
    # Create log dir
    args = parse_arguments()
    save_dir = args.log_dir
    logger = Logger(output_dir=save_dir)
    title='Learning Curve'
    x, y = logger.load_results(["EpLen", "EpRet"])

    x = cumulative_sum(x)
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()
    if args.save:
        fig = plt.figure(title)
        plt.plot(x, y)
        plt.xlabel('Number of Timesteps')
        plt.ylabel('Rewards')
        plt.title(title + " Smoothed")
        plt.savefig(os.path.join(save_dir, "learning_curve.png"))

if __name__ == '__main__':
    main()