"""
Plot effect k has on:
  -Predictive performance.
  -Deletion efficiency.
  -Training Times.
"""
import os
import argparse
import sys
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
sys.path.insert(0, here + '/../../')

import pandas as pd
import matplotlib.pyplot as plt

from plot_delete_cbg import gini_dataset_dict
from plot_delete_cbg import entropy_dataset_dict


def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return width, height


def main(args):
    print(args)

    # get selected hyperparameters
    dataset_dict = gini_dataset_dict if args.criterion == 'gini' else entropy_dataset_dict

    # matplotlib settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('legend', fontsize=11)
    plt.rc('legend', title_fontsize=15)
    plt.rc('lines', linewidth=2)
    plt.rc('lines', markersize=5)

    # setup figure
    width = 3
    width, height = set_size(width=width * 3, fraction=1, subplots=(1, 2))

    fig, axs = plt.subplots(1, 2, figsize=(width, height * 1.1))

    # get results
    main_fp = os.path.join(args.in_dir, 'results.csv')
    main_df = pd.read_csv(main_fp)

    # extract dataset-specific settings
    metric = dataset_dict[args.dataset][0]
    n_trees = dataset_dict[args.dataset][1]
    max_depth = dataset_dict[args.dataset][2]

    # filter results
    df = main_df[main_df['dataset'] == args.dataset]
    df = df[df['criterion'] == args.criterion]
    df = df[df['subsample_size'] == args.subsample_size]
    df = df[df['n_estimators'] == n_trees]
    df = df[df['max_depth'] == max_depth]
    df = df[df['topd'] == 0]

    # plot preditive performance
    ax = axs[0]
    ax.set_ylabel(r'Test error $\Delta$ (%)')
    ax.errorbar(df['k'],
                y=df['model_{}_mean'.format(metric)] * 100,
                yerr=df['model_{}_sem'.format(metric)] * 100,
                color='k')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel('Test {}'.format(metric.upper() if metric in ['auc', 'ap'] else 'Acc.'))

    # plot deletion efficiency
    ax = axs[1]
    ax.set_ylabel('Speedup vs Naive')
    ax.errorbar(df['k'],
                y=df['model_n_deleted'],
                yerr=df['model_n_deleted_std'],
                color='k')
    ax.set_yscale('log')
    ax.set_xlabel(r'$k$')

    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # remove excess space
    fig.tight_layout()

    # save plots
    os.makedirs(args.out_dir, exist_ok=True)
    fp = os.path.join(args.out_dir, '{}.pdf'.format(args.dataset))
    plt.savefig(fp)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bank_marketing', help='dataset to use for plotting.')
    parser.add_argument('--in_dir', type=str, default='output/delete/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/k/', help='output directory.')
    parser.add_argument('--criterion', type=str, default='gini', help='split criterion.')
    parser.add_argument('--subsample_size', type=int, default=1, help='adversary strength.')
    args = parser.parse_args()
    main(args)
