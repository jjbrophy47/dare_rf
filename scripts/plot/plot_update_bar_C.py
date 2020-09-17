"""
Plot update results with CEDAR results as a clustered bar graph
that fits in one column of a two-column paper.
"""
import os
import argparse
import sys
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
sys.path.insert(0, here + '/../../')
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean

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


def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return width, height


def set_hatches(ax, hatches, n_datasets):
    """
    Set hatches in same way for each dataset.
    """
    bars = ax.patches
    hatch_cnt = 0
    hatch_ndx = 0

    for i in range(len(bars)):

        if hatch_cnt == n_datasets:
            hatch_ndx += 1
            hatch_cnt = 0

        bars[i].set_hatch(hatches[hatch_ndx])
        hatch_cnt += 1


def print_summary_stats(summary_stats, subsample_size):
    """
    Print the min, max, and geometric mean of the efficiency.
    """
    print('\nSummary statistics (Sub-{})'.format(subsample_size))
    for model, values in summary_stats.items():
        min_val = min(values)
        max_val = max(values)
        geo_mean = gmean(values)
        print('[{}]: min = {}, max = {}, geometric mean = {}'.format(model, min_val, max_val, geo_mean))


def organize_results(args, df):
    """
    Put results into dataset clusters.
    """
    results = []
    n_model_std_list = []
    metric_diff_std_list = []
    n_datasets = 0
    summary_stats = defaultdict(list)

    for dataset in args.dataset:

        dataset_n_model_std_list = []
        dataset_metric_diff_std_list = []

        result = {'dataset': dataset}
        metric = dataset_dict[dataset][0]
        n_trees = dataset_dict[dataset][1]
        max_depth = dataset_dict[dataset][2]

        temp1 = df[df['dataset'] == dataset]
        temp1 = temp1[temp1['n_estimators'] == n_trees]
        temp1 = temp1[temp1['max_depth'] == max_depth]

        n_datasets += 1

        # add exact
        exact_df = temp1[temp1['model'] == 'exact']
        result['exact_n_model'] = exact_df['n_model'].values[0]
        dataset_n_model_std_list.append(exact_df['n_model_std'].values[0])
        summary_stats['exact'].append(exact_df['n_model'].values[0])

        # add dart
        for i, topd in enumerate(dataset_dict[dataset][3][:2]):

            if topd == 0:
                result['dart_{}_n_model'.format(i)] = result['exact_n_model']
                result['dart_{}_metric_diff'.format(i)] = 0
                dataset_n_model_std_list.append(exact_df['n_model_std'].values[0])
                dataset_metric_diff_std_list.append(0)
                summary_stats['dart_{}'.format(i)].append(result['exact_n_model'])
            else:
                dart_df = temp1[(temp1['model'] == 'dart') & (temp1['topd'] == topd)]
                result['dart_{}_n_model'.format(i)] = dart_df['n_model'].values[0]
                result['dart_{}_metric_diff'.format(i)] = dart_df['{}_diff_mean'.format(metric)].values[0] * 100
                dataset_n_model_std_list.append(dart_df['n_model_std'].values[0])
                dataset_metric_diff_std_list.append(dart_df['{}_diff_std'.format(metric)].values[0] * 100)
                summary_stats['dart_{}'.format(i)].append(dart_df['n_model'].values[0])

        # add CEDAR
        for i, (epsilon, lmbda) in enumerate(zip(args.epsilon, args.lmbda)):
            cedar_df = temp1[(temp1['model'] == 'cedar') & (temp1['epsilon'] == epsilon) & (temp1['lmbda'] == lmbda)]
            result['cedar_{}_n_model'.format(i)] = cedar_df['n_model'].values[0]
            result['cedar_{}_metric_diff'.format(i)] = cedar_df['{}_diff_mean'.format(metric)].values[0] * 100
            dataset_n_model_std_list.append(cedar_df['n_model_std'].values[0])
            dataset_metric_diff_std_list.append(cedar_df['{}_diff_std'.format(metric)].values[0] * 100)
            summary_stats['cedar_{}'.format(i)].append(cedar_df['n_model'].values[0])

        n_model_std_list += dataset_n_model_std_list + dataset_n_model_std_list
        metric_diff_std_list += dataset_metric_diff_std_list + dataset_metric_diff_std_list

        results.append(result)
    res_df = pd.DataFrame(results)

    return res_df, n_model_std_list, metric_diff_std_list, n_datasets, summary_stats


def main(args):
    print(args)

    # matplotlib settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=15)
    plt.rc('axes', titlesize=15)
    plt.rc('legend', fontsize=11)
    plt.rc('legend', title_fontsize=9)
    plt.rc('lines', linewidth=2)
    plt.rc('lines', markersize=5)

    # setup figure
    width = 5
    width, height = set_size(width=width * 3, fraction=1, subplots=(2, 3))
    fig, axs = plt.subplots(2, 1, figsize=(width, height), sharex=True)

    tol_list = ['0.1%', '0.25%', '0.5%', '1.0%']
    tol_list = tol_list[:2]

    cedar_settings = list(zip(args.epsilon, args.lmbda))

    labels = ['D-DART']
    labels += ['R-DART (tol={})'.format(tol) for tol in tol_list]
    labels += [r'CEDR ($\epsilon$={}, $\lambda$={:.0e})'.format(ep, lm) for ep, lm in cedar_settings]

    assert args.subsample_size == 1
    titles = ['Efficiency Using the Random Adversary'.format(args.subsample_size),
              'Absolute Test Error Increase Relative to D-DART (lower is better)']

    colors = ['0.0', '1.0', '0.75', '0.5', '0.25']
    colors += ['0.5', '0.25']

    hatches = ['', '', '', 'o', '*']

    # get results
    main_fp = os.path.join(args.in_dir, 'results.csv')
    main_df = pd.read_csv(main_fp)

    # filter results
    df = main_df[main_df['operation'] == args.operation]
    df = df[df['criterion'] == args.criterion]
    df = df[df['subsample_size'] == args.subsample_size]

    res_df, n_model_std, metric_diff_std_list, n_datasets, summary_stats = organize_results(args, df)

    print_summary_stats(summary_stats, subsample_size=args.subsample_size)

    n_model_y = ['exact_n_model']
    n_model_y += ['dart_{}_n_model'.format(i) for i in range(len(tol_list))]
    n_model_y += ['cedar_{}_n_model'.format(i) for i in range(len(cedar_settings))]
    metric_diff_y = ['dart_{}_metric_diff'.format(i) for i in range(len(tol_list))]
    metric_diff_y += ['cedar_{}_metric_diff'.format(i) for i in range(len(cedar_settings))]

    n_methods = 1 + len(tol_list) + len(cedar_settings)

    n_model_yerr = np.reshape(n_model_std, (n_methods, 2, n_datasets), order='F')
    metric_diff_yerr = np.reshape(metric_diff_std_list, (n_methods - 1, 2, n_datasets), order='F')

    res_df.plot(x='dataset', y=n_model_y, yerr=n_model_yerr, kind='bar',
                color=colors, ax=axs[0], edgecolor='k', linewidth=0.5, capsize=2)

    res_df.plot(x='dataset', y=metric_diff_y, yerr=metric_diff_yerr, kind='bar',
                color=colors[1:], ax=axs[1], edgecolor='k', linewidth=0.5, capsize=2)

    set_hatches(axs[0], hatches, n_datasets)
    set_hatches(axs[1], hatches[1:], n_datasets)

    axs[0].set_yscale('log')
    axs[0].grid(which='major', axis='y')
    axs[0].set_axisbelow(True)
    axs[0].set_ylim(bottom=1)
    axs[0].set_ylabel('Speedup vs Naive')
    axs[0].set_yticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    axs[0].set_title(titles[0], loc='left')
    leg = axs[0].legend(labels=labels, ncol=3, framealpha=1.0, loc='upper right')

    x_labels = [label.get_text().replace('_', ' ').title() if i % 2 == 0 else \
                '\n' + label.get_text().replace('_', ' ').title() for i, label in
                enumerate(axs[1].xaxis.get_majorticklabels())]
    axs[1].set_xticklabels(x_labels, rotation=0)
    axs[1].set_xlabel('Dataset')
    axs[1].set_ylabel(r'Test error $\Delta$ (%)')
    axs[1].grid(which='major', axis='y')
    axs[1].set_axisbelow(True)
    axs[1].get_legend().remove()
    axs[1].set_title(titles[1], loc='left')

    # Get the bounding box of the original legend
    bb = leg.get_bbox_to_anchor().inverse_transformed(axs[0].transAxes)

    # Change to location of the legend.
    yOffset = 0.35
    bb.y0 += yOffset
    bb.y1 += yOffset
    leg.set_bbox_to_anchor(bb, transform=axs[0].transAxes)

    out_dir = os.path.join(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    fig.tight_layout()
    fp = os.path.join(out_dir, '{}_{}_sub{}_cedar.pdf'.format(args.criterion,
                                                              args.operation,
                                                              args.subsample_size))
    plt.savefig(fp)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='+', help='datasets to use for plotting',
                        default=['surgical', 'vaccine', 'adult', 'bank_marketing', 'flight_delays',
                                 'diabetes', 'olympics', 'census', 'credit_card', 'synthetic',
                                 'higgs'])
    parser.add_argument('--in_dir', type=str, default='output/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/update_bar_C/', help='output directory.')

    parser.add_argument('--criterion', type=str, default='gini', help='split criterion.')
    parser.add_argument('--operation', type=str, default='delete', help='add or delete.')
    parser.add_argument('--subsample_size', type=int, default=1, help='adversary strength.')

    parser.add_argument('--epsilon', type=float, nargs='+', default=[10, 10], help='epsilon.')
    parser.add_argument('--lmbda', type=float, nargs='+', default=[0.1, 0.0001], help='lmbda.')
    args = parser.parse_args()
    main(args)
