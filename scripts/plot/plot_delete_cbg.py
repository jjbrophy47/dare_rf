"""
Plot delete results as clustered bar graphs:
    1) Speedup vs Naive (Random adversary)
    2) Speedup vs Naive (Worst-of-1000 adversary)
    3) Predictive Performance Difference BEFORE deletion
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

N_TOL = 5

# selected hyperparameters
gini_dataset_dict = {'surgical': ('acc', 100, 20, 25, [0, 0, 1, 2, 4]),
                     'vaccine': ('acc', 50, 20, 5, [0, 5, 7, 11, 14]),
                     'adult': ('acc', 50, 20, 5, [0, 10, 13, 14, 16]),
                     'bank_marketing': ('auc', 100, 20, 25, [0, 6, 9, 12, 14]),
                     'flight_delays': ('auc', 250, 20, 25, [0, 1, 3, 5, 10]),
                     'diabetes': ('acc', 250, 20, 5, [0, 7, 10, 12, 15]),
                     'no_show': ('auc', 250, 20, 10, [0, 1, 3, 6, 10]),
                     'olympics': ('auc', 250, 20, 5, [0, 0, 1, 2, 3]),
                     'census': ('auc', 100, 20, 25, [0, 6, 9, 12, 16]),
                     'credit_card': ('ap', 250, 20, 5, [0, 5, 8, 14, 17]),
                     'ctr': ('auc', 100, 10, 50, [0, 2, 3, 4, 6]),
                     'twitter': ('auc', 100, 20, 5, [0, 2, 4, 7, 11]),
                     'synthetic': ('acc', 50, 20, 10, [0, 0, 2, 3, 5]),
                     'higgs': ('acc', 50, 20, 10, [0, 1, 3, 6, 9])
                     }

entropy_dataset_dict = {'surgical': ('acc', 100, 20, 50, [0, 1, 1, 2, 4]),
                        'vaccine': ('acc', 250, 20, 5, [0, 6, 9, 11, 15]),
                        'adult': ('acc', 50, 20, 5, [0, 9, 12, 14, 15]),
                        'bank_marketing': ('auc', 100, 10, 10, [0, 1, 1, 3, 4]),
                        'flight_delays': ('auc', 250, 20, 50, [0, 1, 3, 5, 10]),
                        'diabetes': ('acc', 100, 20, 5, [0, 4, 10, 11, 14]),
                        'no_show': ('auc', 250, 20, 10, [0, 1, 3, 6, 9]),
                        'olympics': ('auc', 250, 20, 5, [0, 0, 1, 2, 4]),
                        'census': ('auc', 100, 20, 25, [0, 5, 8, 11, 15]),
                        'credit_card': ('ap', 250, 10, 25, [0, 1, 2, 3, 4]),
                        'ctr': ('auc', 100, 10, 25, [0, 2, 3, 4, 6]),
                        'twitter': ('auc', 100, 20, 5, [0, 3, 5, 8, 11]),
                        'synthetic': ('acc', 50, 20, 10, [0, 1, 2, 3, 6]),
                        'higgs': ('acc', 50, 20, 10, [0, 0, 2, 5, 8])
                        }


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

    # get selected hyperparameters
    dataset_dict = gini_dataset_dict if args.criterion == 'gini' else entropy_dataset_dict

    # result containers
    results = []
    model_n_deleted_std_list = []
    metric_diff_sem_list = []
    n_datasets = 0
    summary_stats = defaultdict(list)

    for dataset in args.dataset:

        dataset_model_n_deleted_std_list = []
        dataset_metric_diff_sem_list = []

        result = {'dataset': dataset}
        metric = dataset_dict[dataset][0]
        n_trees = dataset_dict[dataset][1]
        max_depth = dataset_dict[dataset][2]
        k = dataset_dict[dataset][3]

        # filter using hyperparameters
        temp1 = df[df['dataset'] == dataset]
        temp1 = temp1[temp1['n_estimators'] == n_trees]
        temp1 = temp1[temp1['max_depth'] == max_depth]
        temp1 = temp1[temp1['k'] == k]

        # skip dataset
        if len(temp1) < 10:
            print('{}, skip'.format(dataset))
            continue
        else:
            print('{}'.format(dataset))

        n_datasets += 1

        # add dart
        for i, topd in enumerate(dataset_dict[dataset][4][:N_TOL]):

            # get dataset specifics
            temp2 = temp1[temp1['topd'] == topd]

            # add speedup and predictive performance means to the result object
            result['{}_model_n_deleted'.format(i)] = temp2['model_n_deleted'].values[0]
            result['{}_metric_diff'.format(i)] = temp2['{}_diff_mean'.format(metric)].values[0] * 100

            # add standard errors to their respective lists
            dataset_model_n_deleted_std_list.append(temp2['model_n_deleted_std'].values[0])
            dataset_metric_diff_sem_list.append(temp2['{}_diff_sem'.format(metric)].values[0] * 100)

            # add speedup to the summary statistics
            summary_stats['{}'.format(i)].append(temp2['model_n_deleted'].values[0])

        model_n_deleted_std_list += dataset_model_n_deleted_std_list + dataset_model_n_deleted_std_list
        metric_diff_sem_list += dataset_metric_diff_sem_list + dataset_metric_diff_sem_list

        results.append(result)

    results_df = pd.DataFrame(results)

    return results_df, model_n_deleted_std_list, metric_diff_sem_list, n_datasets, summary_stats


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
    ax1.set_ylim(1, 1e6)
    ax2 = fig.add_subplot(312, sharey=ax1, sharex=ax1)
    ax3 = fig.add_subplot(313, sharex=ax1)

    tol_list = ['0.0%', '0.1%', '0.25%', '0.5%', '1.0%']
    tol_list = tol_list[:N_TOL]

    labels = ['G-DaRE']
    labels += ['R-DaRE (tol={})'.format(tol) for tol in tol_list[1:]]

    titles = ['Deletion Efficiency Using the Random Adversary (higher is better)',
              'Deletion Efficiency Using the Worst-of-{} Adversary (higher is better)',
              'Test Error Increase Relative to G-DaRE RF (lower is better)']

    colors = ['0.0', '1.0', '0.8', '0.6', '0.4']
    colors += ['0.5', '0.75']

    hatches = ['\\', '-', '^', 'o', '*']

    # get results
    main_fp = os.path.join(args.in_dir, 'results.csv')
    main_df = pd.read_csv(main_fp)

    # filter results
    df = main_df[main_df['criterion'] == args.criterion]
    sub1_df = df[df['subsample_size'] == 1]
    subX_df = df[df['subsample_size'] == args.subsample_size]

    # setup columns
    n_model_y = ['{}_model_n_deleted'.format(i) for i in range(len(tol_list))]
    metric_diff_y = ['{}_metric_diff'.format(i) for i in range(len(tol_list))]

    # get results
    print('\nCompiling random adversary results:')
    sub1_res_df, sub1_n_model_std, metric_diff_sem_list, sub1_n_datasets, sub1_stats = organize_results(args, sub1_df)
    print('\nCompiling worst-of-1000 adversary results:')
    subX_res_df, subX_n_model_std, _, subX_n_datasets, subX_stats = organize_results(args, subX_df)

    # print summary results
    print_summary_stats(sub1_stats, subsample_size=1)
    print_summary_stats(subX_stats, subsample_size=args.subsample_size)

    n_methods = len(tol_list)

    # get standard errors
    sub1_n_model_yerr = np.reshape(sub1_n_model_std, (n_methods, 2, sub1_n_datasets), order='F')
    subX_n_model_yerr = np.reshape(subX_n_model_std, (n_methods, 2, sub1_n_datasets), order='F')
    metric_diff_yerr = np.reshape(metric_diff_sem_list, (n_methods, 2, sub1_n_datasets), order='F')

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
    set_hatches(ax3, hatches, sub1_n_datasets)

    # plot-specific settings
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
    x_labels = [x.upper() if 'Ctr' in x else x for x in x_labels]
    ax3.set_xticklabels(x_labels, rotation=0)
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel(r'Test error $\Delta$ (%)')
    ax3.grid(which='major', axis='y')
    ax3.set_axisbelow(True)
    ax3.get_legend().remove()
    ax3.set_title(titles[2], loc='left')

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
    fp = os.path.join(out_dir, '{}_{}.pdf'.format(args.criterion,
                                                  args.subsample_size))
    plt.savefig(fp)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='+', help='datasets to use for plotting',
                        default=['surgical', 'vaccine', 'adult', 'bank_marketing', 'flight_delays',
                                 'diabetes', 'no_show', 'olympics', 'census', 'credit_card', 'ctr',
                                 'synthetic', 'twitter', 'higgs'])
    parser.add_argument('--in_dir', type=str, default='output/delete/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/delete_cbg/', help='output directory.')
    parser.add_argument('--criterion', type=str, default='gini', help='split criterion.')
    parser.add_argument('--subsample_size', type=int, default=1000, help='adversary strength.')
    args = parser.parse_args()
    main(args)
