"""
Plot update results with CEDAR results as a clustered bar graph.
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

N_TOL = 2

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
        max_features = dataset_dict[dataset][3]

        temp1 = df[df['dataset'] == dataset]
        temp1 = temp1[temp1['n_estimators'] == n_trees]
        temp1 = temp1[temp1['max_depth'] == max_depth]
        temp1 = temp1[temp1['max_features'] == max_features]

        n_datasets += 1

        # add exact
        exact_df = temp1[temp1['model'] == 'exact']
        result['exact_n_model'] = exact_df['n_model'].values[0]
        dataset_n_model_std_list.append(exact_df['n_model_std'].values[0])
        summary_stats['exact'].append(exact_df['n_model'].values[0])

        # add dart
        for i, topd in enumerate(dataset_dict[dataset][4][:N_TOL]):

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
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    plt.rc('axes', labelsize=24)
    plt.rc('axes', titlesize=24)
    plt.rc('legend', fontsize=20)
    plt.rc('legend', title_fontsize=9)
    plt.rc('lines', linewidth=2)
    plt.rc('lines', markersize=5)

    # setup figure
    width = 5
    width, height = set_size(width=width * 5, fraction=1, subplots=(2, 3))
    fig = plt.figure(figsize=(width, height * 1.25))

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharey=ax1, sharex=ax1)
    ax3 = fig.add_subplot(313, sharex=ax1)

    tol_list = ['0.1%', '0.25%', '0.5%', '1.0%']
    tol_list = tol_list[:N_TOL]

    cedar_settings = list(zip(args.epsilon, args.lmbda))

    labels = ['D-DART']
    labels += ['R-DART (tol={})'.format(tol) for tol in tol_list]
    labels += [r'CEDAR ($\epsilon$={}, $\lambda$={:.0e})'.format(ep, lm) for ep, lm in cedar_settings]

    titles = ['Efficiency Using the Random Adversary (higher is better)',
              'Efficiency Using the Worst-of-{} Adversary (higher is better)',
              'Difference in Efficiency Between the Random and Worst-of-{} Adversaries (lower is better)',
              'Test Error Increase Relative to D-DART (lower is better)']

    colors = ['0.0', '1.0', '0.8', '0.6', '0.4']
    colors += ['0.5', '0.75']

    hatches = ['', '', '', 'o', '*']

    # get results
    main_fp = os.path.join(args.in_dir, 'results.csv')
    main_df = pd.read_csv(main_fp)

    # filter results
    df = main_df[main_df['operation'] == args.operation]
    df = df[df['criterion'] == args.criterion]
    sub1_df = df[df['subsample_size'] == 1]
    subX_df = df[df['subsample_size'] == args.subsample_size]

    # setup columns
    n_model_y = ['exact_n_model']
    n_model_y += ['dart_{}_n_model'.format(i) for i in range(len(tol_list))]
    n_model_y += ['cedar_{}_n_model'.format(i) for i in range(len(cedar_settings))]
    metric_diff_y = ['dart_{}_metric_diff'.format(i) for i in range(len(tol_list))]
    metric_diff_y += ['cedar_{}_metric_diff'.format(i) for i in range(len(cedar_settings))]

    # get results
    sub1_res_df, sub1_n_model_std, metric_diff_std_list, sub1_n_datasets, sub1_stats = organize_results(args, sub1_df)
    subX_res_df, subX_n_model_std, _, subX_n_datasets, subX_stats = organize_results(args, subX_df)

    # print summary results
    print_summary_stats(sub1_stats, subsample_size=1)
    print_summary_stats(subX_stats, subsample_size=args.subsample_size)

    n_methods = 1 + len(tol_list) + len(cedar_settings)

    # get standard errors
    sub1_n_model_yerr = np.reshape(sub1_n_model_std, (n_methods, 2, sub1_n_datasets), order='F')
    subX_n_model_yerr = np.reshape(subX_n_model_std, (n_methods, 2, sub1_n_datasets), order='F')
    metric_diff_yerr = np.reshape(metric_diff_std_list, (n_methods - 1, 2, sub1_n_datasets), order='F')

    # plot results
    sub1_res_df.plot(x='dataset', y=n_model_y, yerr=sub1_n_model_yerr, kind='bar',
                     color=colors, ax=ax1, edgecolor='k', linewidth=0.5, capsize=2)

    subX_res_df.plot(x='dataset', y=n_model_y, yerr=subX_n_model_yerr, kind='bar',
                       color=colors, ax=ax2, edgecolor='k', linewidth=0.5, capsize=2)

    sub1_res_df.plot(x='dataset', y=metric_diff_y, yerr=metric_diff_yerr, kind='bar',
                     color=colors[1:], ax=ax3, edgecolor='k', linewidth=0.5, capsize=2)

    # set hatches
    set_hatches(ax1, hatches, sub1_n_datasets)
    set_hatches(ax2, hatches, sub1_n_datasets)
    set_hatches(ax3, hatches[1:], sub1_n_datasets)

    # plot specific settings
    ax1.set_yscale('log')
    ax1.grid(which='major', axis='y')
    ax1.set_axisbelow(True)
    ax1.set_ylim(bottom=1)
    ax1.set_ylabel('Speedup vs Naive')
    ax1.set_yticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    ax1.set_title(titles[0], loc='left')
    leg = ax1.legend(labels=labels, ncol=3, framealpha=1.0, loc='upper right')

    ax2.set_yscale('log')
    ax2.grid(which='major', axis='y')
    ax2.set_axisbelow(True)
    ax2.set_ylim(bottom=1)
    ax2.set_title(titles[1].format(args.subsample_size), loc='left')
    ax2.set_ylabel('Speedup vs Naive')
    ax2.set_yticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    ax2.get_legend().remove()

    x_labels = [label.get_text().replace('_', ' ').title() if i % 2 == 0 else \
                '\n' + label.get_text().replace('_', ' ').title() for i, label in
                enumerate(ax3.xaxis.get_majorticklabels())]
    ax3.set_xticklabels(x_labels, rotation=0)
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel(r'Test error $\Delta$ (%)')
    ax3.grid(which='major', axis='y')
    ax3.set_axisbelow(True)
    ax3.get_legend().remove()
    ax3.set_title(titles[3], loc='left')

    # Get the bounding box of the original legend
    bb = leg.get_bbox_to_anchor().inverse_transformed(ax1.transAxes)

    # Change to location of the legend.
    yOffset = 0.375
    bb.y0 += yOffset
    bb.y1 += yOffset
    leg.set_bbox_to_anchor(bb, transform=ax1.transAxes)

    out_dir = os.path.join(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    fig.tight_layout()
    fp = os.path.join(out_dir, '{}_{}_cedar.pdf'.format(args.criterion,
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
    parser.add_argument('--in_dir', type=str, default='output/update/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/update_bar_C/', help='output directory.')

    parser.add_argument('--criterion', type=str, default='gini', help='split criterion.')
    parser.add_argument('--operation', type=str, default='deletion', help='addition or deletion.')
    parser.add_argument('--subsample_size', type=int, default=1000, help='adversary strength.')

    parser.add_argument('--epsilon', type=float, nargs='+', default=[10, 10], help='epsilon.')
    parser.add_argument('--lmbda', type=float, nargs='+', default=[0.1, 0.0001], help='lmbda.')
    args = parser.parse_args()
    main(args)
