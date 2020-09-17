"""
Plot update results as four clustered bar graphs:
    1) Sub-1 efficiency
    2) Sub-1000 efficiency
    3) Sub-1 - Sub-1000 efficiency
    4) Test error increase BEFORE addition/deletion
"""
import os
import argparse
import sys
from collections import defaultdict
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
sys.path.insert(0, here + '/../../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean

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

dataset_dict = {'surgical': ('acc', 100, 10, [0, 3, 4, 6]),
                'vaccine': ('acc', 250, 20, [0, 8, 12, 15]),
                'adult': ('acc', 250, 20, [11, 12, 14, 16]),
                'bank_marketing': ('auc', 100, 10, [2, 4, 5, 7]),
                'flight_delays': ('auc', 250, 20, [2, 5, 10, 17]),
                'diabetes': ('acc', 250, 20, [2, 6, 11, 17]),
                'olympics': ('auc', 250, 20, [0, 0, 1, 2]),
                'census': ('auc', 250, 20, [3, 6, 9, 15]),
                'credit_card': ('ap', 250, 20, [2, 4, 6, 13]),
                'synthetic': ('acc', 250, 20, [1, 3, 5, 7]),
                'higgs': ('acc', 100, 10, [0, 1, 2, 3])}


def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return width, height


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

        if len(temp1) < 4:
            continue

        n_datasets += 1

        # add exact
        exact_df = temp1[temp1['model'] == 'exact']
        result['exact_n_model'] = exact_df['n_model'].values[0]
        dataset_n_model_std_list.append(exact_df['n_model_std'].values[0])
        summary_stats['exact'].append(exact_df['n_model'].values[0])

        # add dart
        for i, topd in enumerate(dataset_dict[dataset][3]):

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
        if args.cedar:
            for i, (epsilon, lmbda) in enumerate(zip(args.epsilon, args.lmbda)):
                cedar_df = temp1[temp1['model'] == 'cedar']
                cedar_df = cedar_df[cedar_df['epsilon'] == epsilon]
                cedar_df = cedar_df[cedar_df['lmbda'] == lmbda]

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


def main(args):
    print(args)

    if False:

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

        # setup figure
        width = 2.5
        width, height = set_size(width=width * 3, fraction=1, subplots=(2, 3))
        fig, axs = plt.subplots(4, 1, figsize=(width, height * 3), sharex=True)

    else:

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
        fig, axs = plt.subplots(3, 1, figsize=(width, height * 1.25), sharex=True)

    tol_list = ['0.1%', '0.25%', '0.5%', '1.0%']
    labels = ['D-DART']
    labels += ['R-DART (tol={})'.format(tol) for tol in tol_list]
    if args.cedar:
        labels += [r'CEDR ($\epsilon=10$)']
    colors = ['0.0', '1.0', '0.75', '0.5', '0.25', '0.1']

    titles = ['Efficiency Using the Random Adversary (higher is better)',
              'Efficiency Using the Worst-of-{} Adversary (higher is better)',
              'Difference in Efficiency Between the Random and Worst-of-{} Adversaries (lower is better)',
              'Absolute Test Error Increase Relative to D-DART (lower is better)']

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
    if args.cedar:
        n_model_y += ['cedar_{}_n_model'.format(i) for i in range(len(args.lmbda))]

    metric_diff_y = ['dart_{}_metric_diff'.format(i) for i in range(len(tol_list))]
    if args.cedar:
        metric_diff_y += ['cedar_{}_metric_diff'.format(i) for i in range(len(args.lmbda))]

    # get results
    sub1_res_df, sub1_n_model_std, metric_diff_std_list, sub1_n_datasets, sub1_stats = organize_results(args, sub1_df)
    subX_res_df, subX_n_model_std, _, subX_n_datasets, subX_stats = organize_results(args, subX_df)

    # print summary statistics
    print_summary_stats(sub1_stats, subsample_size=1)
    print_summary_stats(subX_stats, subsample_size=args.subsample_size)

    # compute difference between adversaries
    subd_res_df = sub1_res_df.copy()
    for col in n_model_y:
        subd_res_df[col] = sub1_res_df[col] / subX_res_df[col]
    # subd_n_model_std = [np.mean([x, y]) for x, y in zip(sub1_n_model_std, subX_n_model_std)]

    assert sub1_n_datasets == subX_n_datasets

    print('\n{}'.format(sub1_res_df.head(5)))
    print(subX_res_df.head(5))
    print(subd_res_df.head(5))

    n_methods = len(labels)

    sub1_n_model_yerr = np.reshape(sub1_n_model_std, (n_methods, 2, sub1_n_datasets), order='F')
    subX_n_model_yerr = np.reshape(subX_n_model_std, (n_methods, 2, sub1_n_datasets), order='F')
    metric_diff_yerr = np.reshape(metric_diff_std_list, (n_methods - 1, 2, sub1_n_datasets), order='F')

    sub1_res_df.plot(x='dataset', y=n_model_y, yerr=sub1_n_model_yerr, kind='bar',
                     color=colors, ax=axs[0], edgecolor='k', linewidth=0.5, capsize=2)

    subX_res_df.plot(x='dataset', y=n_model_y, yerr=subX_n_model_yerr, kind='bar',
                       color=colors, ax=axs[1], edgecolor='k', linewidth=0.5, capsize=2)

    # subd_res_df.plot(x='dataset', y=n_model_y, yerr=None, kind='bar',
    #                  color=colors, ax=axs[2], edgecolor='k', linewidth=0.5, capsize=2)

    sub1_res_df.plot(x='dataset', y=metric_diff_y, yerr=metric_diff_yerr, kind='bar',
                     color=colors[1:], ax=axs[2], edgecolor='k', linewidth=0.5, capsize=2)

    axs[0].set_yscale('log')
    axs[0].grid(which='major', axis='y')
    axs[0].set_axisbelow(True)
    axs[0].set_ylim(bottom=1)
    axs[0].set_title(titles[0], loc='left')
    axs[0].set_ylabel('Speedup vs Naive')
    axs[0].set_yticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    leg = axs[0].legend(labels=labels, ncol=3, framealpha=1.0, loc='upper right')

    axs[1].set_yscale('log')
    axs[1].grid(which='major', axis='y')
    axs[1].set_axisbelow(True)
    axs[1].set_ylim(bottom=1)
    axs[1].set_title(titles[1].format(args.subsample_size), loc='left')
    axs[1].set_ylabel('Speedup vs Naive')
    axs[1].set_yticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    axs[1].get_legend().remove()

    # axs[2].set_yscale('log')
    # axs[2].grid(which='major', axis='y')
    # axs[2].set_axisbelow(True)
    # axs[2].set_ylim(bottom=1)
    # axs[2].set_title(titles[2].format(args.subsample_size), loc='left')
    # axs[2].set_ylabel('Speedup vs Naive')
    # axs[2].set_ylim(bottom=1)
    # axs[2].set_yticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    # axs[2].get_legend().remove()

    x_labels = [label.get_text().replace('_', ' ').title() if i % 2 == 0 else \
                '\n' + label.get_text().replace('_', ' ').title() for i, label in
                enumerate(axs[2].xaxis.get_majorticklabels())]
    axs[2].set_xticklabels(x_labels, rotation=0)
    axs[2].set_title(titles[3], loc='left')
    axs[2].set_xlabel('Dataset')
    axs[2].set_ylabel(r'Test error $\Delta$ (%)')
    axs[2].grid(which='major', axis='y')
    axs[2].set_axisbelow(True)
    axs[2].get_legend().remove()

    # Get the bounding box of the original legend
    bb = leg.get_bbox_to_anchor().inverse_transformed(axs[0].transAxes)

    # Change to location of the legend.
    yOffset = 0.375
    bb.y0 += yOffset
    bb.y1 += yOffset
    leg.set_bbox_to_anchor(bb, transform=axs[0].transAxes)

    out_dir = os.path.join(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    fig.tight_layout()
    fp = os.path.join(out_dir, '{}_{}.pdf'.format(args.criterion, args.operation,
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
    parser.add_argument('--out_dir', type=str, default='output/plots/update_bar_D/', help='output directory.')

    parser.add_argument('--criterion', type=str, default='gini', help='split criterion.')
    parser.add_argument('--operation', type=str, default='delete', help='add or delete.')
    parser.add_argument('--subsample_size', type=int, default=1000, help='adversary strength.')

    parser.add_argument('--cedar', action='store_true', default=False, help='include CEDR results.')
    parser.add_argument('--epsilon', type=float, nargs='+', default=[10], help='epsilon.')
    parser.add_argument('--lmbda', type=float, nargs='+', default=[0.1], help='lmbda.')
    args = parser.parse_args()
    main(args)
