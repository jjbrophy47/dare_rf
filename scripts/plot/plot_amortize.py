"""
Plot results of amortize experiment (deletion) for a single dataset.
Only use colors when absolutely necessary.
"""
import os
import sys
import argparse
from collections import Counter
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
sys.path.insert(0, here + '/../../')

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from scripts.print import print_util


def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return width, height


def get_limits(num_list):
    """
    Returns the lower and upper bounds of the smallest
    and biggest numbers.
    """
    min_num, max_num = min(num_list), max(num_list)

    # lower bound
    i = -10
    while 10 ** i < min_num:
        i += 1
    lb = 10 ** (i - 1)

    # upper bound
    i = -10
    while 10 ** i < max_num:
        i += 1
    ub = 10 ** i

    return lb, ub


def _get_retrainings(args, r):
    """
    Get mean number of retrainings.
    """
    c = Counter()
    for rs in args.rs:
        types = r[rs]['type']
        depths = r[rs]['depth']
        retrain_indices = np.where(types == 2)[0]
        retrain_depths = depths[retrain_indices]
        temp_counter = Counter(retrain_depths)
        c.update(temp_counter)

    depths = sorted(c.keys())
    counts = [c[d] / args.repeats for d in depths]
    return depths, counts


def main(args):

    dataset = args.dataset
    metric = args.metric

    adversaries = ['random', 'root']
    adversary_color = ['red', 'purple', 'green']
    adversary_marker = ['o', '*', '^']

    models = ['naive', 'exact', 'cedar']
    model_linestyle = ['-', '--']
    model_labels = ['Exact', 'CeDAR']

    metric_labels = {'acc': 'Accuracy', 'auc': 'AUROC'}

    # get results
    r = {}

    for adversary in adversaries:
        r[adversary] = {model: {} for model in models}

        naive = print_util.get_results(args, dataset, adversary, 'naive')
        exact = print_util.get_results(args, dataset, adversary, 'exact')
        cedar = print_util.get_results(args, dataset, adversary, 'cedar_ep{}'.format(args.epsilon))

        n_train, _ = print_util.get_mean1d(args, naive, 'n_train', as_int=True)
        n_features, _ = print_util.get_mean1d(args, naive, 'n_features', as_int=True)
        max_depth, _ = print_util.get_mean1d(args, naive, 'max_depth', as_int=True)

        n_trees = 1
        max_features = 'N/A'
        if args.model_type == 'forest':
            n_trees, _ = print_util.get_mean1d(args, naive, 'n_estimators', as_int=True)
            max_features = print_util.get_max_features(args, naive, 'max_features')

        s = '\nDataset: {} ({:,} instances, {:,} features)   Trees: {:,}   '
        s += 'Max depth: {}   Max features: {}'
        if args.verbose:
            print(s.format(dataset, n_train, n_features, n_trees, max_depth, max_features))

        amortize, amortize_sem = print_util.get_mean_amortize(args, naive)
        s = '\n[Naive] amortized: {:.5f}s'
        if args.verbose:
            print(s.format(amortize))
        r[adversary]['naive']['amortize'] = amortize
        r[adversary]['naive']['amortize_sem'] = amortize

        amortize, amortize_sem = print_util.get_mean_amortize(args, exact)
        exact_depths, exact_counts = _get_retrainings(args, exact)
        scores, scores_sem = print_util.get_mean(args, exact, args.metric)
        if args.verbose:
            print('\n[Exact] amortized: {:.5f}s'.format(amortize))
            print('[Exact] retrains: {}'.format(list(zip(exact_depths, exact_counts))))
            print('[Exact] scores: {}'.format(scores))
        r[adversary]['exact']['amortize'] = amortize
        r[adversary]['exact']['amortize_sem'] = amortize_sem
        r[adversary]['exact']['depths'] = exact_depths
        r[adversary]['exact']['counts'] = exact_counts
        r[adversary]['exact']['scores'] = scores
        r[adversary]['exact']['scores_sem'] = scores_sem

        amortize, amortize_sem = print_util.get_mean_amortize(args, cedar)
        cedar_depths, cedar_counts = _get_retrainings(args, cedar)
        scores, scores_sem = print_util.get_mean(args, cedar, args.metric)
        if args.verbose:
            print('\n[CeDAR] amortized: {:.5f}s'.format(amortize))
            print('[CeDAR] retrains: {}'.format(list(zip(cedar_depths, cedar_counts))))
            print('[CeDAR] scores: {}'.format(scores))
        r[adversary]['cedar']['amortize'] = amortize
        r[adversary]['cedar']['amortize_sem'] = amortize_sem
        r[adversary]['cedar']['depths'] = cedar_depths
        r[adversary]['cedar']['counts'] = cedar_counts
        r[adversary]['cedar']['scores'] = scores
        r[adversary]['cedar']['scores_sem'] = scores_sem

    # matplotlib settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('axes', labelsize=13)
    plt.rc('axes', titlesize=13)
    plt.rc('legend', fontsize=9)
    plt.rc('legend', title_fontsize=9)
    plt.rc('lines', linewidth=2)
    plt.rc('lines', markersize=5)

    # inches
    width = 5.5
    width, height = set_size(width=width * 2, fraction=1, subplots=(1, 2))

    fig, axs = plt.subplots(1, 3, figsize=(width, height))
    title_str = 'Dataset: {} ({:,} instances, {:,} features)   Trees: {:,}   Max depth: {}   Max features: {}'
    if args.verbose:
        print(title_str.format(dataset, n_train, n_features, n_trees, max_depth, max_features))

    lb, ub = 100, -100

    for i, adversary in enumerate(adversaries):
        if args.verbose:
            print('\nAdversary: {}'.format(adversary))

        # plot amortized times
        labels = ['Naive', 'Exact\n' r'$\epsilon=0$' '\n' r'$\lambda=\infty$',
                  'CeDAR\n' r'$\epsilon={}$'.format(args.epsilon)]
        order = np.arange(len(labels))
        order = order + i / 10
        times = np.array([r[adversary][model]['amortize'] for model in models])

        temp_lb, temp_ub = get_limits(times)
        lb = min(lb, temp_lb)
        ub = max(ub, temp_ub)

        ax = axs[0]
        ax.scatter(order, times, color=adversary_color[i], marker=adversary_marker[i],
                   label=adversary.capitalize())
        ax.set_xticks(order - i / 20)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Time (s)')
        ax.set_title('Train once + delete 10%')
        ax.set_yscale('log')
        ax.set_ylim(lb, ub)
        ax.grid(True, axis='y')
        ax.legend(title='Adversary')

        # plot retrains
        ax = axs[1]

        for j, model in enumerate(['exact', 'cedar']):
            depth_list = r[adversary][model]['depths']
            count_list = r[adversary][model]['counts']

            label = '{}: {}'.format(adversary.capitalize(), model_labels[j])
            ax.plot(depth_list, count_list,
                    marker=adversary_marker[i],
                    color=adversary_color[i],
                    linestyle=model_linestyle[j],
                    label=label)

        ax.set_xlabel('Tree depth')
        ax.set_ylabel('# retrains')
        ax.set_title('Retrains across all trees')
        ax.legend(title='Adversary: Model', ncol=1, handlelength=3)

        # plot performance
        ax = axs[2]

        for j, model in enumerate(['exact', 'cedar']):
            label = '{}: {}'.format(adversary.capitalize(), model.capitalize())

            scores = r[adversary][model]['scores']
            percentages = np.arange(len(scores)) / 100 * 100

            ax.plot(percentages, scores, color=adversary_color[i],
                    marker=adversary_marker[i], linestyle=model_linestyle[j], label=label)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        ax.set_xlabel('% data deleted')
        ax.set_ylabel('Test {}'.format(metric_labels[metric]))
        ax.set_title('Predictive performance')

    os.makedirs(args.out_dir, exist_ok=True)

    if args.pdf:
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, '{}.pdf'.format(dataset)))

    else:
        fig.suptitle(title_str.format(dataset, n_train, n_features, n_trees, max_depth, max_features))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(args.out_dir, '{}.png'.format(dataset)))

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='surgical', help='dataset to plot.')
    parser.add_argument('--in_dir', type=str, default='output/amortize/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/amortize/', help='output directory.')

    parser.add_argument('--metric', type=str, default='auc', help='predictive performance metric.')
    parser.add_argument('--criterion', type=str, default='gini', help='split criterion.')
    parser.add_argument('--model_type', type=str, default='forest', help='stump, tree, or forest.')

    parser.add_argument('--epsilon', type=str, default='0.1', help='epsilon value.')
    parser.add_argument('--pdf', action='store_true', default=False, help='save a pdf of this plot.')

    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='random states.')
    parser.add_argument('--repeats', type=int, default=5, help='number of experiments.')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level.')
    args = parser.parse_args()
    print(args)
    main(args)


# External API
class Args:
    in_dir = 'output/amortize/'
    out_dir = 'plots/amortize/'
    dataset = 'surgical'
    rs = 1
    repeats = 1
    metric = 'auc'
    model_type = 'forest'
    epsilon = 0.1
    pdf = False
    verbose = 0
