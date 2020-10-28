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
    output = []
    for idx in range(len(values)):
        output.append(np.mean(values[:idx+1][-window:]))

    return output

def cumulative_sum(x):
    x = np.array(x)
    output = []
    for i in range(len(x)):
        output.append(x[:i+1].sum())
    return output

def standardise_graph(x1, y1, x2, y2):
    for count, i in enumerate(x2):
        if i >= x1[-1]:
            x2 = x2[:count]
            y2 = y2[:count]
    return x2, y2

def standardise_lengths(x, max_length):
    standardised_x = []
    for i in x:
        standardised_x.append(i + [i[-1]]*(max_length-len(i)))
    
    return standardised_x

def plot_results(env, agent, show_each_trial=False):
    save_dir = os.path.join("Model_Weights", env, agent)
    logger = Logger(output_dir=save_dir, load=True)
    EpLen_list, EpRet_list = logger.load_all_results(["EpLen", "EpRet"])
    Ep_Returns, Ep_Lengths = [], []
    max_length = len(EpLen_list[0])
    max_idx = 0
    for idx, (EpLen, EpRet) in enumerate(zip(EpLen_list, EpRet_list)):
        EpLen = cumulative_sum(EpLen)
        EpRet = moving_average(EpRet, window=50)
        if show_each_trial:
            plt.plot(x, y, label=f"trial: {idx+1}")
        if len(EpLen) > max_length:
            max_length = len(EpLen)
            max_idx = idx
        Ep_Returns.append(EpRet)
        Ep_Lengths.append(EpLen)

    EpLen = Ep_Lengths[max_idx]
    Ep_Returns = np.array(standardise_lengths(Ep_Returns, max_length)).T

    ret_mean = []
    ret_std = []
    for ep_ret in Ep_Returns:
        ret_mean.append(ep_ret.mean())
        ret_std.append(ep_ret.std()) 

    ret_mean = np.array(ret_mean)
    ret_std = np.array(moving_average(ret_std, 50))
    
    if not show_each_trial:
        plt.plot(EpLen, ret_mean, label=f"{agent}")
        plt.fill_between(EpLen, ret_mean-ret_std, ret_mean+ret_std, alpha=0.2)

def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--log_dir', type=str, default='Model_Weights\CartPoleContinuousBulletEnv-v0\ddpg', help='path to log directory')
    parser.add_argument('--env', type=str, required=True, help='path to log directory')
    parser.add_argument('--agent', type=str, help='path to log directory')
    parser.add_argument('--save', action='store_true', help='if true, save the plot to log directory')
    parser.add_argument('--baseline', action='store_true', help='if true, plot the results alongside stable_baselines3 trained model')
    parser.add_argument('--compare', action='store_true', help='if true, plot the results alongside every other algorithm trained on the same environment')
    return parser.parse_args()

def main():
    # Create log dir
    args = parse_arguments()
    title='Learning Curve'
    fig = plt.figure(title)
    if args.agent is not None:
        plot_results(args.env, args.agent)
    if args.compare:
        agents = os.listdir(os.path.join("Model_Weights", args.env))
        if args.agent is not None:
            agents.remove(args.agent)
        for agent in agents:
            plot_results(args.env, agent)

    elif args.baseline:
        log_dir = os.path.join("Stable_Baselines", "logs", os.path.sep.join(args.log_dir.split(os.path.sep)[1:]))
        from stable_baselines3.common.results_plotter import load_results, ts2xy
        x2, y2 = ts2xy(load_results(log_dir), 'timesteps')
        y2 = moving_average(y2, window=50)
        # Truncate x
        x2 = x2[len(x2) - len(y2):]
        x2, y2 = standardise_graph(x, y, x2, y2)

        plt.plot(x2, y2, label="Stable_Baselines3 implementation")
    
    plt.legend()
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    if args.save:
        fname = "comparison.png" if args.compare else "learning_curve.png"
        save_dir = os.path.join("Model_Weights", args.env)
        plt.savefig(os.path.join(save_dir, fname))
    plt.show()

if __name__ == '__main__':
    main()