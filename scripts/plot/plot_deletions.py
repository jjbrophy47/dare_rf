"""
Plot delete_until retrain results.
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
# from scipy.stats import sem


def get_n_plots(args):
    """
    Return size of the figure based on the number of datasets.
    """
    n_datasets = len(args.dataset)
    n_rows = int((n_datasets / 4)) + 1
    n_cols = min(n_datasets, 4)
    return n_rows, n_cols


def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return width, height


def main(args):

    colors = ['k', 'k']
    lines = ['-', '--']
    markers = ['x', '^']

    # matplotlib settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=22)
    plt.rc('axes', titlesize=22)
    plt.rc('legend', fontsize=15)
    plt.rc('legend', title_fontsize=15)
    plt.rc('lines', linewidth=3)
    plt.rc('lines', markersize=10)

    width = 5.5
    n_rows, n_cols = get_n_plots(args)
    width, height = set_size(width=width * n_cols, fraction=1, subplots=(n_rows, n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(width, height * 1.25), sharey=True)
    axs = axs.flatten()

    for i, dataset in enumerate(args.dataset):
        ax = axs[i]

        for j, adversary in enumerate(args.adversary):
            print('\n{}'.format(adversary.capitalize()))

            # get results
            r = {}
            for rs in args.rs:
                fp = os.path.join(args.in_dir, dataset, args.model_type, args.criterion,
                                  adversary, 'rs{}'.format(rs), 'results.npy')
                r[rs] = np.load(fp, allow_pickle=True)[()]

            n_train = r[1]['n_train']
            n_features = r[1]['n_features']

            n_deletions, n_deletions_sem = print_util.get_mean(args, r, name='n_deletions')
            if not args.no_pct:
                n_deletions = [x / n_train * 100 for x in n_deletions]
                n_deletions_sem = [x / n_train * 100 for x in n_deletions_sem]
            train_time, _ = print_util.get_mean1d(args, r, name='train_time')
            lmbda, _ = print_util.get_mean1d(args, r, name='lmbda')
            epsilons = r[1]['epsilon']
            theoretical_n_deletions = [(ep / lmbda) * n_train for ep in epsilons]

            out_str = '\n{} ({:,} instances, {:,} features), train_time: {:.5f}s'
            print(out_str.format(dataset, n_train, n_features, train_time))
            print('lmbda: {}'.format(lmbda))
            print('epsilons: {}'.format(epsilons))
            print('n_deletions: {}'.format(n_deletions))
            print('theoretical deletions: {}'.format(theoretical_n_deletions))

            label = adversary.capitalize()
            # ax.plot(epsilons, n_deletions, color=colors[j], linestyle=lines[j],
            #         marker=markers[j], label=label)
            ax.errorbar(epsilons, n_deletions, yerr=n_deletions_sem, color=colors[j],
                        linestyle=lines[j], marker=markers[j], label=label)
            ax.set_xscale('log')
            ax.set_xlabel(r'$\epsilon$')
            ax.grid(True)

            ylabel = '% deleted'
            if args.no_pct:
                ax.set_yscale('log')
                ylabel = '# deletions'

        if i == 0:
            ax.legend(title='Adversary', handlelength=3)
            ax.set_ylabel(ylabel)

        if args.png:
            title_str = '{} ({:,} train instances, {:,} features), {}'
            ax.set_title(title_str.format(dataset, n_train, n_features, args.model_type))

        else:
            ax.set_title(dataset)

    out_dir = os.path.join(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dataset_label = '_'.join(d for d in args.dataset)

    if args.png:
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '{}.png'.format(dataset_label)), bbox_inches='tight')

    else:
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '{}.pdf'.format(dataset_label)), bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='+', default=['surgical', 'mfc20'], help='datasets to show.')
    parser.add_argument('--in_dir', type=str, default='output/delete_until_retrain', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/delete_until_retrain', help='output directory.')

    parser.add_argument('--adversary', type=str, nargs='+', default=['random', 'root'], help='adversary to show.')
    parser.add_argument('--model_type', type=str, nargs='+', default='forest', help='models to show.')
    parser.add_argument('--criterion', type=str, default='gini', help='split criterion.')

    parser.add_argument('--png', action='store_true', default=False, help='save a png of this plot.')
    parser.add_argument('--no_pct', action='store_true', default=False, help='do not show percentages.')

    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='random state.')
    args = parser.parse_args()
    main(args)
