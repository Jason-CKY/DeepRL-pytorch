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
    parser.add_argument('--compare', action='store_true', help='if true, plot the results alongside stable_baselines3 trained model')
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
    if args.compare:
        log_dir = os.path.join("Stable_Baselines", "logs", os.path.sep.join(args.log_dir.split(os.path.sep)[1:]))
        from stable_baselines3.common.results_plotter import load_results, ts2xy
        x2, y2 = ts2xy(load_results(log_dir), 'timesteps')
        y2 = moving_average(y2, window=50)
        # Truncate x
        x2 = x2[len(x2) - len(y2):]
        plt.plot(x2, y2)
        
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    if args.save:
        plt.savefig(os.path.join(save_dir, "learning_curve.png"))
    plt.show()

if __name__ == '__main__':
    main()