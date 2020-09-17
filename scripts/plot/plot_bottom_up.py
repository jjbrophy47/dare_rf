"""
Plot predictive performance of model that adds nodes
in a bottom-up fashion and compare to DART and Random.
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
    width = 2.5
    width, height = set_size(width=width * 3, fraction=1, subplots=(1, 1))
    fig, ax = plt.subplots(figsize=(width, height))

    colors = ['0.0', '1.0', '0.75', '0.5', '0.25']

    labels = ['Greedy', 'Random', 'Bottom Only']
    but_error = [0.214, 0.252, 0.199, 0.101, 0.298, 0.378,
                 0.273, 0.101, 0.249, 0.177, 0.362]
    random_error = [0.218, 0.231, 0.209, 0.097, 0.297, 0.438,
                    0.245, 0.09, 0.27, 0.171, 0.329]
    dart_error = [0.204, 0.21, 0.179, 0.093, 0.276, 0.369,
                  0.174, 0.078, 0.184, 0.099, 0.316]

    res_dict = {'Dataset': args.dataset,
                'Greedy': np.array(dart_error) * 100,
                'Bottom Only': np.array(but_error) * 100,
                'Random': np.array(random_error) * 100}
    df = pd.DataFrame(res_dict)

    df.plot(x='Dataset', y=labels, kind='barh', color=colors, ax=ax,
            edgecolor='k', linewidth=0.5)

    ax.grid(which='major', axis='x')
    ax.set_axisbelow(True)
    ax.set_xlabel('Test error (%)')
    ax.legend(labels=labels, ncol=1, framealpha=1.0)

    y_labels = [label.get_text().replace('_', ' ').title() for i, label in
                enumerate(ax.yaxis.get_majorticklabels())]
    ax.set_yticklabels(y_labels, rotation=0)

    os.makedirs(args.out_dir, exist_ok=True)

    fig.tight_layout()
    fp = os.path.join(args.out_dir, 'bottom_up.pdf')
    plt.savefig(fp)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='+', help='datasets to use for plotting',
                        default=['surgical', 'vaccine', 'adult', 'bank_marketing', 'flight_delays',
                                 'diabetes', 'olympics', 'census', 'credit_card', 'synthetic',
                                 'higgs'])
    parser.add_argument('--in_dir', type=str, default='output/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/bottom_up/', help='output directory.')
    parser.add_argument('--criterion', type=str, default='gini', help='split criterion.')
    args = parser.parse_args()
    main(args)
