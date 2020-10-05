"""
Plot detailed results for exact alternate methods.
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

dataset_dict = {'surgical': ('acc', 250, 20, [2, 13, 18, 19]),
                'vaccine': ('acc', 250, 10, [1, 4, 5, 7]),
                'adult': ('acc', 10, 20, [15, 17, 18, 19]),
                'bank_marketing': ('auc', 100, 10, [9, 9, 9, 9]),
                'flight_delays': ('auc', 250, 20, [4, 9, 11, 15]),
                'diabetes': ('acc', 250, 20, [6, 11, 15, 19]),
                'olympics': ('auc', 250, 20, [0, 1, 2, 4]),
                'census': ('auc', 250, 20, [9, 13, 15, 18]),
                'credit_card': ('ap', 250, 20, [5, 5, 10, 12]),
                'synthetic': ('acc', 250, 20, [2, 4, 6, 9]),
                'higgs': ('acc', 100, 10, [1, 2, 4, 5])}

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


def main(args):
    print(args)

    # matplotlib settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('axes', labelsize=13)
    plt.rc('axes', titlesize=13)
    plt.rc('legend', fontsize=11)
    plt.rc('legend', title_fontsize=9)
    plt.rc('lines', linewidth=2)
    plt.rc('lines', markersize=5)

    n_datasets = len(args.dataset)

    # setup figure
    width = 3
    width, height = set_size(width=width * 3, fraction=1, subplots=(n_datasets, 2))
    fig, axs = plt.subplots(n_datasets, 2, figsize=(width, height * 1.5))

    if axs.ndim == 1:
        axs = np.expand_dims(axs, axis=0)

    shape_size = 10
    shape_list = ['o', '^', 's', 'd']

    tol_list = ['0.1%', '0.25%', '0.5%', '1.0%']

    lines = [':', '-.', '-']
    lines += lines

    # get results
    main_fp = os.path.join(args.in_dir, 'results.csv')
    main_df = pd.read_csv(main_fp)

    for i, dataset in enumerate(args.dataset):

        metric = dataset_dict[dataset][0]
        n_trees = dataset_dict[dataset][1]
        max_depth = dataset_dict[dataset][2]
        max_features = dataset_dict[dataset][3]

        # filter results
        df = main_df[main_df['dataset'] == dataset]
        df = df[df['operation'] == args.operation]
        df = df[df['criterion'] == args.criterion]
        df = df[df['subsample_size'] == args.subsample_size]
        df = df[df['max_features'] == max_features]
        # df['avg_deletion_time'] = df['allotted_time'] / df['n_model']

        # settings in question
        exp_df = df[df['n_estimators'] == n_trees]
        exp_df = exp_df[exp_df['max_depth'] == max_depth]

        # models
        exact_df = exp_df[exp_df['model'] == 'exact']
        dart_df = exp_df[exp_df['model'] == 'dart']
        dart_df = pd.concat([exact_df, dart_df])
        alt_df = df[(df['model'] == 'exact') &
                    (df['n_estimators'] != n_trees) &
                    (df['max_depth'] != max_depth)]

        # plot efficiency
        ax = axs[i][0]
        ax.set_ylabel('Average deletion time (s)')
        ax.errorbar(dart_df['topd'], dart_df['deletion_time_mean'], yerr=dart_df['deletion_time_std'],
                    label='R-DART', color='k')
        for tol, topd, shape in zip(tol_list, dataset_dict[dataset][4], shape_list):
            x = dart_df['topd'].iloc[topd]
            y = dart_df['deletion_time_mean'].iloc[topd]
            ax.plot(x, y, 'k{}'.format(shape), label='tol={}'.format(tol), ms=shape_size)
        ax.axhline(exact_df['deletion_time_mean'].values[0], color='k', linestyle='--', label='D-DART')
        for j, row in enumerate(alt_df.itertuples(index=False)):
            ax.axhline(row.deletion_time_mean, label=r'$T={}$, $D={}$'.format(row.n_estimators, row.max_depth),
                       linestyle=lines[j], color='k')
        ax.set_yscale('log')
        ax.set_title('Efficiency (lower is better)')
        if i == n_datasets - 1:
            ax.set_xlabel(r'$topd$')
        handles, labels = ax.get_legend_handles_labels()

        # plot utility
        ax = axs[i][1]
        ax.set_ylabel('Test {}'.format(metric.upper()))
        ax.errorbar(dart_df['topd'], dart_df['{}_mean'.format(metric)],
                    yerr=dart_df['{}_diff_std'.format(metric)], color='k')
        for tol, topd, shape in zip(tol_list, dataset_dict[dataset][4], shape_list):
            x = dart_df['topd'].iloc[topd]
            y = dart_df['{}_mean'.format(metric)].iloc[topd]
            ax.plot(x, y, 'k{}'.format(shape), label='tol={}'.format(tol), ms=shape_size)
        ax.axhline(exact_df['{}_mean'.format(metric)].values[0], color='k', linestyle='--')
        for j in range(len(alt_df)):
            ax.axhline(alt_df.iloc[j]['{}_mean'.format(metric)], linestyle=lines[j], color='k')
        ax.set_title('Utility (higher is better)')
        if i == n_datasets - 1:
            ax.set_xlabel(r'$topd$')

    os.makedirs(args.out_dir, exist_ok=True)

    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), ncol=4, loc='lower center', frameon=True)
    fig.tight_layout(rect=[0, 0.125, 1, 1])
    fp = os.path.join(args.out_dir, '{}_sub{}_{}.pdf'.format(args.operation,
                                                             args.subsample_size,
                                                             dataset))
    plt.savefig(fp)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='+', help='datasets to use for plotting', default=['credit_card'])
    parser.add_argument('--in_dir', type=str, default='output/update/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/update_detail_C/', help='output directory.')

    parser.add_argument('--criterion', type=str, default='gini', help='split criterion.')
    parser.add_argument('--operation', type=str, default='deletion', help='addition or deletion.')
    parser.add_argument('--subsample_size', type=int, default=1, help='adversary strength.')
    args = parser.parse_args()
    main(args)
