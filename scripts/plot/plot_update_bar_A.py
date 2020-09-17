"""
Plot update results as a clustered bar graph.
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

        # add dart
        for i, topd in enumerate(dataset_dict[dataset][3]):

            if topd == 0:
                result['dart_{}_n_model'.format(i)] = result['exact_n_model']
                result['dart_{}_metric_diff'.format(i)] = 0
                dataset_n_model_std_list.append(exact_df['n_model_std'].values[0])
                dataset_metric_diff_std_list.append(0)
            else:
                dart_df = temp1[(temp1['model'] == 'dart') & (temp1['topd'] == topd)]
                result['dart_{}_n_model'.format(i)] = dart_df['n_model'].values[0]
                result['dart_{}_metric_diff'.format(i)] = dart_df['{}_diff_mean'.format(metric)].values[0] * 100
                dataset_n_model_std_list.append(dart_df['n_model_std'].values[0])
                dataset_metric_diff_std_list.append(dart_df['{}_diff_std'.format(metric)].values[0] * 100)

        n_model_std_list += dataset_n_model_std_list + dataset_n_model_std_list
        metric_diff_std_list += dataset_metric_diff_std_list + dataset_metric_diff_std_list

        results.append(result)
    res_df = pd.DataFrame(results)

    return res_df, n_model_std_list, metric_diff_std_list, n_datasets


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

    # setup figure
    width = 5
    width, height = set_size(width=width * 3, fraction=1, subplots=(2, 3))
    fig, axs = plt.subplots(2, 1, figsize=(width, height * 0.93), sharex=True)

    tol_list = ['0.1%', '0.25%', '0.5%', '1.0%']
    labels = ['Exact'] + ['DART (tol={})'.format(tol) for tol in tol_list]
    colors = ['0.0', '1.0', '0.75', '0.5', '0.25']

    # get results
    main_fp = os.path.join(args.in_dir, 'results.csv')
    main_df = pd.read_csv(main_fp)

    # filter results
    df = main_df[main_df['operation'] == args.operation]
    df = df[df['criterion'] == args.criterion]
    df = df[df['subsample_size'] == args.subsample_size]

    print(df, df.columns)

    res_df, n_model_std, metric_diff_std_list, n_datasets = organize_results(args, df)

    print(res_df)

    n_model_y = ['exact_n_model'] + ['dart_{}_n_model'.format(i) for i in range(len(tol_list))]
    metric_diff_y = ['dart_{}_metric_diff'.format(i) for i in range(len(tol_list))]

    n_model_yerr = np.reshape(n_model_std, (5, 2, n_datasets), order='F')
    metric_diff_yerr = np.reshape(metric_diff_std_list, (4, 2, n_datasets), order='F')

    res_df.plot(x='dataset', y=n_model_y, yerr=n_model_yerr, kind='bar',
                color=colors, ax=axs[0], edgecolor='k', linewidth=0.5, capsize=2)

    res_df.plot(x='dataset', y=metric_diff_y, yerr=metric_diff_yerr, kind='bar',
                color=colors[1:], ax=axs[1], edgecolor='k', linewidth=0.5, capsize=2)

    axs[0].set_yscale('log')
    axs[0].grid(which='major', axis='y')
    axs[0].set_axisbelow(True)
    axs[0].set_ylim(bottom=1)
    axs[0].set_ylabel('Speedup vs Naive')
    axs[0].set_yticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    leg = axs[0].legend(labels=labels, ncol=3, framealpha=1.0)

    x_labels = [label.get_text() if i % 2 == 0 else '\n' + label.get_text() for i, label in
                enumerate(axs[1].xaxis.get_majorticklabels())]
    axs[1].set_xticklabels(x_labels, rotation=0)
    axs[1].set_xlabel('Dataset')
    axs[1].set_ylabel(r'Test error $\Delta$ (%)')
    axs[1].grid(which='major', axis='y')
    axs[1].set_axisbelow(True)
    axs[1].get_legend().remove()
    axs[1].set_ylim(-0.25, 2.35)

    # Get the bounding box of the original legend
    bb = leg.get_bbox_to_anchor().inverse_transformed(axs[0].transAxes)

    # Change to location of the legend.
    yOffset = 0.085
    bb.y0 += yOffset
    bb.y1 += yOffset
    leg.set_bbox_to_anchor(bb, transform=axs[0].transAxes)

    out_dir = os.path.join(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    fig.tight_layout()
    fp = os.path.join(out_dir, '{}_{}_sub{}.pdf'.format(args.criterion, args.operation,
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
    parser.add_argument('--out_dir', type=str, default='output/plots/update_bar_A/', help='output directory.')

    parser.add_argument('--criterion', type=str, default='gini', help='split criterion.')
    parser.add_argument('--operation', type=str, default='delete', help='add or delete.')
    parser.add_argument('--subsample_size', type=int, default=1, help='adversary strength.')
    args = parser.parse_args()
    main(args)
