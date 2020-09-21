import argparse
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title='Learning Curve', save_fig=False):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()
    if save_fig:
        fig = plt.figure(title)
        plt.plot(x, y)
        plt.xlabel('Number of Timesteps')
        plt.ylabel('Rewards')
        plt.title(title + " Smoothed")
        plt.savefig(os.path.join(save_dir, "learning_curve.png"))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True, help='path to log directory')
    parser.add_argument('--save', action='store_true', help='if true, save the plot to log directory')
  
    return parser.parse_args()

def main():
    # Create log dir
    args = parse_arguments()
    plot_results(args.log_dir, save_fig=args.save)

if __name__ == '__main__':
    main()