"""
Plot detailed results for a single dataset using both adversaries.
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

# old DART hyperparameters
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
    plt.rc('axes', labelsize=17)
    plt.rc('axes', titlesize=17)
    plt.rc('legend', fontsize=11)
    plt.rc('legend', title_fontsize=9)
    plt.rc('lines', linewidth=2)
    plt.rc('lines', markersize=5)

    n_rows = len(args.subsample_size)

    # setup figure
    width = 5
    width, height = set_size(width=width * 3, fraction=1, subplots=(n_rows, 3))

    fig = plt.figure(figsize=(width, height * 0.9))
    gs = fig.add_gridspec(2, 3)

    shape_size = 10
    shape_list = ['o', '^', 's', 'd']

    adversaries = ['Random', 'Worst-of-1000']
    tol_list = ['0.1%', '0.25%', '0.5%', '1.0%']

    # get results
    main_fp = os.path.join(args.in_dir, 'results.csv')
    main_df = pd.read_csv(main_fp)

    retrain_fp = os.path.join(args.in_dir, 'retrain.csv')
    retrain_df = pd.read_csv(retrain_fp)

    for i, subsample_size in enumerate(args.subsample_size):

        metric = dataset_dict[args.dataset][0]
        n_trees = dataset_dict[args.dataset][1]
        max_depth = dataset_dict[args.dataset][2]
        max_features = dataset_dict[args.dataset][3]

        # filter results
        df = main_df[main_df['dataset'] == args.dataset]
        df = df[df['operation'] == args.operation]
        df = df[df['criterion'] == args.criterion]
        df = df[df['subsample_size'] == subsample_size]
        df = df[df['n_estimators'] == n_trees]
        df = df[df['max_depth'] == max_depth]
        df = df[df['max_features'] == max_features]

        exact_df = df[df['model'] == 'exact']
        dart_df = df[df['model'] == 'dart']

        dart_df = pd.concat([exact_df, dart_df])

        # plot efficiency
        if i == 0:
            ax = fig.add_subplot(gs[i, 0])
            prev_efficiency_ax = ax
        else:
            ax = fig.add_subplot(gs[i, 0], sharey=prev_efficiency_ax)
        ax.set_ylabel('({})\nSpeedup vs Naive'.format(adversaries[i], subsample_size))
        ax.errorbar(dart_df['topd'], dart_df['n_model'], yerr=dart_df['n_model_std'], label='R-DART', color='k')
        for tol, topd, shape in zip(tol_list, dataset_dict[args.dataset][4], shape_list):
            x = dart_df['topd'].iloc[topd]
            y = dart_df['n_model'].iloc[topd]
            ax.plot(x, y, 'k{}'.format(shape), label='tol={}'.format(tol), ms=shape_size)
        ax.axhline(exact_df['n_model'].values[0], color='k', linestyle='--', label='D-DART')
        ax.set_yscale('log')
        if i == 0:
            ax.legend(ncol=2, frameon=False)
            ax.set_title('Efficiency')
        elif i == n_rows - 1:
            ax.set_xlabel(r'$topd$')

        # plot utility
        if subsample_size == 1:
            ax = fig.add_subplot(gs[:, 1])
            ax.set_ylabel(r'Test error $\Delta$ (%)')
            ax.errorbar(dart_df['topd'], dart_df['{}_diff_mean'.format(metric)] * 100,
                        yerr=dart_df['{}_diff_std'.format(metric)] * 100, color='k')
            for tol, topd, shape in zip(tol_list, dataset_dict[args.dataset][4], shape_list):
                x = dart_df['topd'].iloc[topd]
                y = dart_df['{}_diff_mean'.format(metric)].iloc[topd] * 100
                ax.plot(x, y, 'k{}'.format(shape), label='tol={}'.format(tol), ms=shape_size)
            ax.axhline(0, color='k', linestyle='--')
            if i == 0:
                ax.set_title('Utility')
            ax.set_xlabel(r'$topd$')

        # plot retrains
        if i == 0:
            ax = fig.add_subplot(gs[i, 2])
            prev_retrain_ax = ax
        else:
            ax = fig.add_subplot(gs[i, 2], sharey=prev_retrain_ax)
        ax.set_ylabel('No. retrains')
        for j, row in enumerate(dart_df[1:].itertuples(index=False)):
            linestyle = '-' if j < 10 else '--'
            temp = retrain_df[retrain_df['id'] == row.id]
            retrains = temp.iloc[0].values[1:]
            depths = np.arange(retrains.shape[0])
            ax.plot(depths[:max_depth], retrains[:max_depth], linestyle=linestyle,
                    label=r'$topd={}$'.format(row.topd))
        temp = retrain_df[retrain_df['id'].isin(exact_df['id'])]
        retrains = temp.iloc[0].values[1:]
        depths = np.arange(retrains.shape[0])
        ax.plot(depths[:max_depth], retrains[:max_depth], 'k--')
        if i == 0:
            ax.set_title('Retrains')
        if i == n_rows - 1:
            ax.set_xlabel('Retrain depth')
        if max_depth in [10, 20]:
            handles, labels = ax.get_legend_handles_labels()

    os.makedirs(args.out_dir, exist_ok=True)

    # set retrain legend
    y_loc = 0.965 if max_depth == 20 else 0.800
    fig.legend(handles, labels, bbox_to_anchor=(0.995, y_loc), ncol=1)

    fig.tight_layout(rect=[0, 0, 0.90, 1])
    fp = os.path.join(args.out_dir, '{}_{}.pdf'.format(args.operation,
                                                       args.dataset))
    plt.savefig(fp)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='credit_card', help='dataset to use for plotting.')
    parser.add_argument('--in_dir', type=str, default='output/update/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/update_detail_A/', help='output directory.')

    parser.add_argument('--criterion', type=str, default='gini', help='split criterion.')
    parser.add_argument('--operation', type=str, default='deletion', help='addition or deletion.')
    parser.add_argument('--subsample_size', type=int, nargs='+', default=[1, 1000], help='adversary strength.')
    args = parser.parse_args()
    main(args)
