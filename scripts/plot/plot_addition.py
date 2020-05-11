"""
Plot results of amortize experiment (addition) for a single dataset.
"""
import os
import argparse
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


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


def main(args):

    dataset = args.dataset
    metric = args.metric
    model_type = args.model_type

    adversaries = ['random', 'root']
    adversary_color = ['red', 'purple', 'green']
    adversary_marker = ['o', '*', '^']

    models = ['naive', 'exact', 'cedar']
    model_linestyle = ['-', '--']
    model_labels = ['Exact', 'CeDAR']

    metric_labels = {'acc': 'Accuracy', 'auc': 'AUC'}

    # get results
    r = {}
    for adversary in adversaries:
        r[adversary] = {}
        for model in models:
            fname = 'cedar_ep{}'.format(args.epsilon) if model == 'cedar' else model
            fp = os.path.join(args.in_dir, dataset, model_type, adversary,
                              'rs{}'.format(args.rs), '{}.npy'.format(fname))
            r[adversary][model] = np.load(fp, allow_pickle=True)[()]

    n_train = r['random']['naive']['n_train']
    n_features = r['random']['naive']['n_features']

    n_trees = r['random']['cedar']['n_estimators']
    max_depth = r['random']['cedar']['max_depth']
    max_features = r['random']['cedar']['max_features']

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    title_str = 'Dataset: {} ({:,} instances, {:,} features)   Trees: {:,}   Max depth: {}   Max features: {:,}'
    fig.suptitle(title_str.format(dataset, n_train, n_features, n_trees, max_depth, max_features))

    print(title_str.format(dataset, n_train, n_features, n_trees, max_depth, max_features))

    for i, adversary in enumerate(adversaries):
        print('\nAdversary: {}'.format(adversary))

        # plot amortized times
        mean_times = [r[adversary][model]['time'].mean() for model in models]
        train_times = [r[adversary][model].get('train_time') for model in models]
        lmbda = int(r[adversary]['cedar']['lmbda'])

        print('epsilon: {}, lmbda: {:,}'.format(args.epsilon, lmbda))
        for model, train_time, amortized_time in zip(models, train_times, mean_times):
            print('[{}] amortized: {:.5f}s'.format(model, amortized_time))

        labels = ['Naive', 'Exact\n' r'$\epsilon=0$' '\n' r'$\lambda=\infty$',
                  'CeDAR\n' r'$\epsilon={}$' '\n' r'$\lambda={:.1e}$'.format(args.epsilon, lmbda)]
        order = np.arange(len(labels))
        order = order + i / 10

        lb, ub = get_limits(mean_times)

        ax = axs[0]
        ax.scatter(order, mean_times, color=adversary_color[i], marker=adversary_marker[i],
                   label=adversary.capitalize())
        ax.set_xticks(order - i / 20)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Amortized runtime (sec)')
        ax.set_title('Train once + {:,} deletions (10%)'.format(len(r[adversary][model]['time']) - 1))
        ax.set_yscale('log')
        ax.set_ylim(lb, ub)
        ax.grid(True, axis='y')
        ax.legend(title='Adversary')

        # plot performance
        ax = axs[2]

        for j, model in enumerate(['exact', 'cedar']):
            label = '{}: {}'.format(adversary.capitalize(), model.capitalize())

            performance = r[adversary][model][metric]
            percentages = np.arange(len(performance)) / 100 * 100

            ax.plot(percentages, performance, color=adversary_color[i],
                    marker=adversary_marker[i], linestyle=model_linestyle[j], label=label)

        ax.set_xlabel('% data deleted')
        ax.set_ylabel('Test {}'.format(metric_labels[metric]))
        ax.set_title('Predictive performance')

        # plot retrains
        ax = axs[1]

        for j, model in enumerate(['exact', 'cedar']):
            types = r[adversary][model]['type']
            depths = r[adversary][model]['depth']

            retrain_depths = depths[np.where(types == 2)]

            if len(retrain_depths) > 0:
                retrain_depth_counter = Counter(retrain_depths)
                depth_count_list = list(retrain_depth_counter.items())
                depth_count_list.sort(key=lambda x: x[0])
                depth_list, count_list = zip(*depth_count_list)

                label = '{}: {}'.format(adversary.capitalize(), model_labels[j])

                ax.plot(depth_list, count_list,
                        marker=adversary_marker[i],
                        color=adversary_color[i],
                        linestyle=model_linestyle[j],
                        label=label)

        ax.set_xlabel('Tree depth')
        ax.set_ylabel('# retrains')
        ax.set_title('Retrains across all trees')
        ax.legend(title='[Adversary]: [Model]', ncol=len(adversaries), fontsize=7)

    os.makedirs(args.out_dir, exist_ok=True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(args.out_dir, '{}.png'.format(dataset)))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='output/addition/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='plots/addition/', help='input directory.')

    parser.add_argument('--dataset', type=str, default='surgical', help='dataset to plot.')
    parser.add_argument('--rs', type=str, default=1, help='experiment random state.')
    parser.add_argument('--metric', type=str, default='auc', help='predictive performance metric.')
    parser.add_argument('--model_type', type=str, default='forest', help='stump, tree, or forest.')
    parser.add_argument('--epsilon', type=str, default='0.1', help='epsilon value.')
    args = parser.parse_args()
    main(args)
