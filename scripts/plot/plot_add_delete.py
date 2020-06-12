"""
Plot cumulative additions and deletions.
"""
import os
import argparse
import sys
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
sys.path.insert(0, here + '/../../')
from scripts.print import print_util

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem


def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return width, height


def _get_times(args, r, n_samples=100):
    """
    Get mean # operations and cumulative times.
    """
    res_cum_time_list_all = []
    res_n_ops_list_all = []

    for rs in args.rs:
        times = r[rs]['time']
        cum_times = np.cumsum(times)
        n_ops = len(times)
        step = int(n_ops / n_samples)

        res_cum_time_list = []
        res_n_ops_list = []
        for j in range(0, n_ops, step):
            res_cum_time_list.append(cum_times[j])
            res_n_ops_list.append(j)

        res_cum_time_list_all.append(res_cum_time_list)
        res_n_ops_list_all.append(res_n_ops_list)

    res_cum_time = np.vstack(res_cum_time_list_all)
    res_n_ops = np.vstack(res_n_ops_list_all)

    res_cum_time_mean = np.mean(res_cum_time, axis=0)
    res_cum_time_sem = sem(res_cum_time, axis=0)

    res_n_ops_mean = np.mean(res_n_ops, axis=0)
    res_n_ops_sem = sem(res_n_ops, axis=0)

    return res_cum_time_mean, res_cum_time_sem, res_n_ops_mean, res_n_ops_sem


def _get_results(args, operation, adversary, method):
    """
    Get results for a single method.
    """
    r = {}
    for rs in args.rs:
        fp = os.path.join(args.in_dir, operation, args.dataset,
                          args.model_type, args.criterion, adversary,
                          'rs{}'.format(rs), '{}.npy'.format(method))
        r[rs] = np.load(fp, allow_pickle=True)[()]
    return r


def single_plot(args, adversary, operation, n_train, fig, ax, colors, lines):
    """
    Plots results for an adversary and operation.
    """
    dataset = args.dataset

    print('naive...')
    naive = _get_results(args, operation, adversary, 'naive')
    n_features, _ = print_util.get_mean1d(args, naive, 'n_features', as_int=True)
    n_trees, _ = print_util.get_mean1d(args, naive, 'n_estimators', as_int=True)
    max_depth, _ = print_util.get_mean1d(args, naive, 'max_depth', as_int=True)
    max_features = print_util.get_max_features(args, naive, 'max_features')

    if not args.pdf:
        fig_s = 'Dataset: {} ({:,} instances, {:,} features)   Trees: {:,}   '
        fig_s += 'Max depth: {}   Max features: {:,}'
        fig.suptitle(fig_s.format(dataset, n_train, n_features, n_trees, max_depth, max_features, adversary))

    # plot naive
    if args.naive:
        naive_cum_time_mean, naive_cum_time_sem, naive_n_ops_mean, naive_n_ops_sem = _get_times(args, naive)
        naive_n_ops_pct = naive_n_ops_mean / n_train * 100

        ax.plot(naive_n_ops_pct, naive_cum_time_mean, color=colors['naive'],
                label='Naive', linestyle=lines['naive'])
        ax.set_yscale('log')

    # plot exact
    print('exact...')
    exact = _get_results(args, operation, adversary, 'exact')
    exact_cum_time_mean, exact_cum_time_sem, exact_n_ops_mean, exact_n_ops_sem = _get_times(args, exact)
    exact_n_ops_pct = exact_n_ops_mean / n_train * 100

    ax.plot(exact_n_ops_pct, exact_cum_time_mean, color=colors['exact'],
            label='Exact', linestyle=lines['exact'])

    # plot CeDAR
    for i, epsilon in enumerate(args.epsilon):
        print('cedar (ep={})...'.format(epsilon))
        cedar = _get_results(args, operation, adversary, 'cedar_ep{}'.format(epsilon))
        cedar_cum_time_mean, cedar_cum_time_sem, cedar_n_ops_mean, cedar_n_ops_sem = _get_times(args, cedar)
        cedar_n_ops_pct = cedar_n_ops_mean / n_train * 100

        ax.plot(cedar_n_ops_pct, cedar_cum_time_mean, color=colors['cedar'][i],
                label='CeDAR ' + r'($\epsilon$={})'.format(epsilon),
                linestyle=lines['cedar'][i])


def main(args):
    print(args)

    colors = {'naive': 'k',
              'exact': 'k',
              'cedar': ['#014422', '#006727', '#248823', '#25A032']}
    lines = {'naive': ':',
             'exact': '--',
             'cedar': ['-', '-', '-', '-']}
    op_map = {'delete': 'amortize', 'add': 'addition'}

    # matplotlib settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('axes', labelsize=17)
    plt.rc('axes', titlesize=17)
    plt.rc('legend', fontsize=11)
    plt.rc('legend', title_fontsize=9)
    plt.rc('lines', linewidth=2)
    plt.rc('lines', markersize=5)

    width = 5
    width, height = set_size(width=width * 3, fraction=1, subplots=(1, 3))
    fig, axs = plt.subplots(1, 4, figsize=(width, height), sharey=True)

    naive = _get_results(args, 'amortize', 'random', 'naive')
    n_train, _ = print_util.get_mean1d(args, naive, 'n_train', as_int=True)

    i = 0
    for operation in args.operation:
        for adversary in args.adversary:
            print('{}: {}'.format(operation, adversary))

            single_plot(args, adversary, op_map[operation], n_train, fig, axs[i], colors, lines)

            op_label = 'Deletion' if operation == 'delete' else 'Addition'
            axs[i].set_title('{} ({})'.format(op_label, adversary))

            if i == 0:
                axs[i].legend()
                axs[i].set_ylabel('Cumulative time (s)')

            xlabel = 'deleted' if operation == 'delete' else 'added'
            axs[i].set_xlabel('% data {}'.format(xlabel))

            i += 1

    os.makedirs(args.out_dir, exist_ok=True)

    if args.pdf:
        fig.tight_layout()
        fp = os.path.join(args.out_dir, '{}.pdf'.format(args.dataset))
        plt.savefig(fp)

    else:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fp = os.path.join(args.out_dir, '{}.png'.format(args.dataset))
        plt.savefig(fp)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='surgical', help='dataset to plot.')
    parser.add_argument('--in_dir', type=str, default='output/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/add_delete/', help='output directory.')

    parser.add_argument('--naive', action='store_true', default=True, help='include naive baseline.')
    parser.add_argument('--operation', type=str, nargs='+', default=['delete', 'add'], help='experiment to show.')
    parser.add_argument('--adversary', type=str, nargs='+', default=['random', 'root'], help='adversary to show.')

    parser.add_argument('--model_type', type=str, default='forest', help='stump, tree, or forest.')
    parser.add_argument('--criterion', type=str, default='gini', help='split criterion.')
    parser.add_argument('--metric', type=str, default='auc', help='predictive performance metric.')
    parser.add_argument('--epsilon', type=str, nargs='+', default=['0.1', '0.25', '0.5', '1.0'], help='epsilon.')

    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='random state.')
    parser.add_argument('--pdf', action='store_true', default=True, help='save a pdf of this plot.')
    args = parser.parse_args()
    main(args)
