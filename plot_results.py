import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from Logger.logger import Logger
import argparse

import pickle
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

def truncate(logger, length=1e6):
    '''
    truncate the EpRet and EpLen to maximum number of time-steps. Used to truncate to 1M timesteps for 
    comparison with other algorithms trained on 1M timesteps.
    Args:
        logger: Logger object loaded from pickle
        length: total timesteps to keep, -1 for full length
    '''
    if length < 0:
        return logger
    total_len = 0
    for i, ep_len in enumerate(logger.logger_list[0]['EpLen']):
        total_len += ep_len
        if total_len > length:
            break    
    logger.logger_list[0]['EpLen'] = logger.logger_list[0]['EpLen'][:i+1]
    logger.logger_list[0]['EpRet'] = logger.logger_list[0]['EpRet'][:i+1]
    return logger

def plot_results(logs_dir, plot_label, show_each_trial=False, window=200, maxlen=-1):
    # save_dir = os.path.join("Model_Weights", env, agent)
    logger = Logger(output_dir=logs_dir, load=True)
    logger = truncate(logger, length=maxlen)
    EpLen_list, EpRet_list = logger.load_all_results(["EpLen", "EpRet"])
    Ep_Returns, Ep_Lengths = [], []
    max_length = len(EpLen_list[0])
    max_idx = 0
    for idx, (EpLen, EpRet) in enumerate(zip(EpLen_list, EpRet_list)):
        EpLen = cumulative_sum(EpLen)
        EpRet = moving_average(EpRet, window=window)
        if show_each_trial:
            plt.plot(EpLen, EpRet, label=f"trial: {idx+1}")
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
        plt.plot(EpLen, ret_mean, label=f"{plot_label}")
        plt.fill_between(EpLen, ret_mean-ret_std, ret_mean+ret_std, alpha=0.2)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help='environment id')
    parser.add_argument('--agent', type=str, help='specify type of agent')
    parser.add_argument('--fname', type=str, default='comparison.png', help='output plot image file name')
    parser.add_argument('--window', type=int, default=50, help="window for moving average")
    parser.add_argument('--maxlen', type=int, default=-1, help="max length for plotting")
    parser.add_argument('--save', action='store_true', help='if true, save the plot to log directory')
    parser.add_argument('--compare', action='store_true', help='if true, plot the results alongside every other algorithm trained on the same environment')
    return parser.parse_args()

def main():
    # Create log dir
    args = parse_arguments()
    title= args.env + ' Learning Curve (Smoothed)'
    fig = plt.figure(title)
    if args.compare:
        path = os.path.join("Model_Weights", args.env)
        plots = glob(os.path.join(path, "**/logs.pickle"), recursive=True)
        plots = [os.path.split(plot)[0] for plot in plots]
        for plot in plots:
            plot_results(logs_dir=plot, plot_label=os.path.split(plot)[1], window=args.window, maxlen=args.maxlen)
    elif args.agent is not None:
        path = os.path.join("Model_Weights", args.env, args.agent)
        plot_results(logs_dir=path, plot_label=args.agent, window=args.window, maxlen=args.maxlen)
   
    plt.legend()
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    if args.save:
        fname = "comparison.png" if args.compare else "learning_curve.png"
        save_dir = os.path.join("Model_Weights", args.env)
        plt.savefig(os.path.join(save_dir, args.fname))
    else:
        plt.show()




if __name__ == '__main__':
    main()