"""
Plot cleaning results.
"""
import os
import argparse
import sys
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
sys.path.insert(0, here + '/../../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_dict = {'surgical': ('acc', 250, 10, 0.25, [0, 2, 4, 6]),
                'vaccine': ('acc', 250, 20, -1.0, [0, 8, 10, 15]),
                'adult': ('acc', 250, 20, -1.0, [11, 12, 14, 16]),
                'bank_marketing': ('auc', 250, 10, 0.25, [3, 4, 6, 7]),
                'flight_delays': ('auc', 250, 20, -1.0, [1, 3, 8, 15]),
                'diabetes': ('acc', 250, 20, -1.0, [3, 7, 10, 16]),
                'olympics': ('auc', 250, 20, 0.25, [0, 1, 1, 3]),
                'census': ('auc', 250, 20, -1.0, [3, 6, 10, 15]),
                'credit_card': ('ap', 250, 10, 0.25, [1, 2, 2, 3]),
                'synthetic': ('acc', 250, 20, 0.25, [2, 3, 5, 7]),
                'higgs': ('acc', 100, 10, 0.25, [0, 1, 2, 4])}


def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return width, height


def str_to_np(arr_str):
    """
    Converts a string representation of an array to a numy array.
    """
    return np.array([float(x) for x in str(arr_str).replace('[', '').replace(']', '').split()])


def main(args):
    print(args)

    # matplotlib settings
    # plt.rc('font', family='serif')
    # plt.rc('xtick', labelsize=24)
    # plt.rc('ytick', labelsize=24)
    # plt.rc('axes', labelsize=24)
    # plt.rc('axes', titlesize=24)
    # plt.rc('legend', fontsize=20)
    # plt.rc('legend', title_fontsize=9)
    # plt.rc('lines', linewidth=2)
    # plt.rc('lines', markersize=5)

    # setup figure
    # width = 5
    # width, height = set_size(width=width, fraction=1, subplots=(1, 1))
    # fix, axs = plt.subplots(1, 2, figsize=(width * 5, height * 1.25))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    labels = ['Random', 'D-DART', 'D-DART loss']

    # get results
    fp = os.path.join(args.in_dir, 'results.csv')
    main_df = pd.read_csv(fp)
    metric = dataset_dict[args.dataset][0]

    # filter results
    df = main_df[main_df['dataset'] == args.dataset]
    df = df[df['criterion'] == args.criterion]

    for i, metric in enumerate(['acc', 'auc']):
        ax = axs[i]

        # plot results
        for j, method in enumerate(args.method):
            temp = df[df['method'] == method]
            percentage = str_to_np(temp['checked_pct'].iloc[0])
            performance = str_to_np(temp.iloc[0][metric])
            performance_std = str_to_np(temp.iloc[0]['{}_std'.format(metric)])
            ax.errorbar(percentage, performance, yerr=performance_std, label=labels[j])

        ax.axhline(temp.iloc[0]['{}_clean'.format(metric)], label='Clean model',
                   color='k', linestyle='--')
        ax.set_title(metric)
        ax.set_xlabel('% train data checked')
        ax.set_ylabel(metric.capitalize())
        ax.legend()

    out_dir = os.path.join(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    fig.tight_layout()
    fp = os.path.join(out_dir, '{}.pdf'.format(args.dataset))
    plt.savefig(fp)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='surgical', help='datasets to use for plotting')
    parser.add_argument('--in_dir', type=str, default='output/cleaning/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/cleaning/', help='output directory.')

    parser.add_argument('--criterion', type=str, default='gini', help='split criterion.')
    parser.add_argument('--method', type=str, nargs='+', default=['random', 'dart', 'dart_loss'], help='method.')
    args = parser.parse_args()
    main(args)
