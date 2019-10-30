"""
This experiment chooses m random instances to delete,
then deletes them individually.
BABC: Binary Attributes Binary Classification.
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
from mulan.trees.babc_tree import BABC_Tree
from utility import data_util


def _fit_delete_refit(X, y, delete_ndx, refit=False, max_depth=4):
    """
    This method first a tree, efficiently deletes the target instance,
    and refits a new tree without the target instance, and
    returns the times for each of these events.
    """
    result = {}

    t1 = BABC_Tree(max_depth=args.max_depth).fit(X, y)
    start = time.time()
    result['delete_type'] = t1.delete(delete_ndx)
    result['delete'] = time.time() - start

    if refit:
        X_new = np.delete(X, delete_ndx, axis=0)
        y_new = np.delete(y, delete_ndx)

        start = time.time()
        t2 = BABC_Tree(max_depth=max_depth).fit(X_new, y_new)
        result['refit'] = time.time() - start
        result['refit_to_delete_ratio'] = result['refit'] / result['delete']
        assert t1.equals(t2)

    return result


def _display_results(results, args, n_samples, n_attributes, n_remove, out_dir='output/bbac/individual_removal'):
    """
    Plot the average time of deletion for each deletion type.
    """
    df = pd.DataFrame(results)

    deletion_sum = df['delete'].sum()
    print('deletion sum: {:.3f}s'.format(deletion_sum))
    print(df[df['delete_type'] == '3'])

    f = plt.figure(figsize=(20, 4))
    ax0 = f.add_subplot(141)
    ax1 = f.add_subplot(142)
    ax2 = f.add_subplot(143, sharey=ax1)
    ax3 = f.add_subplot(144)
    sns.countplot(x='delete_type', data=df, ax=ax0)
    df.boxplot(column='delete', by='delete_type', ax=ax2)
    if not args.no_refit:
        refit_sum = df['refit'].sum()
        print('refit sum: {:.3f}s'.format(refit_sum))
        df.boxplot(column='refit', by='delete_type', ax=ax1)
        df.boxplot(column='refit_to_delete_ratio', by='delete_type', ax=ax3)
        ax1.set_ylabel('seconds')
        ax3.set_ylabel('ratio')

    ax0.set_ylabel('count')
    ax0.set_xlabel('delete_type')
    ax0.set_title('deletion occurrences')
    title_str = 'dataset: {}, samples: {}, attributes: {}, removed: {}\n'
    f.suptitle(title_str.format(args.dataset, n_samples, n_attributes, n_remove))

    out_dir = os.path.join(out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 's{}_a{}_r{}.pdf'.format(n_samples, n_attributes, n_remove)),
                bbox_inches='tight')


def main(args):

    # obtain dataset
    if args.dataset == 'synthetic':

        np.random.seed(args.seed)
        X = np.random.randint(2, size=(args.n_samples, args.n_attributes))

        np.random.seed(args.seed)
        y = np.random.randint(2, size=args.n_samples)

    else:
        X, _, y, _ = data_util.get_data(args.dataset, convert=False)

    # retrieve the indices to remove
    n_samples = X.shape[0]
    n_attributes = X.shape[1]
    n_remove = int(args.remove_frac * n_samples) if args.n_remove is None else args.n_remove
    np.random.seed(args.seed)
    indices_to_delete = np.random.choice(np.arange(n_samples), size=n_remove, replace=False)
    print('n_samples: {}, n_attributes: {}'.format(n_samples, n_attributes))

    # delete instances one at a time and measure the time
    results = []
    for i, ndx in enumerate(indices_to_delete):
        result = _fit_delete_refit(X, y, int(ndx), refit=not args.no_refit, max_depth=args.max_depth)

        if args.verbose > 0:
            if int(0.1 * n_remove) != 0 and i % int(0.1 * n_remove) == 0:
                print('removed: {}'.format(i))

        if args.verbose > 1:
            print('\nDeleting instance {}'.format(ndx))
            print('delete_type: {}'.format(result['delete_type']))
            print('delete: {:.5f}'.format(result['delete']))
            if not args.no_refit:
                print('refit: {:.5f}'.format(result['refit']))
                print('refit-to-delete ratio: {:.3f}'.format(result['refit_to_delete_ratio']))

        results.append(result)

    _display_results(results, args, n_samples, n_attributes, n_remove)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='synthetic', help='dataset to use for the experiment.')
    parser.add_argument('--n_samples', type=int, default=10, help='number of samples to generate.')
    parser.add_argument('--n_attributes', type=int, default=4, help='number of attributes to generate.')
    parser.add_argument('--remove_frac', type=float, default=0.1, help='fraction of instances to delete.')
    parser.add_argument('--n_remove', type=int, default=None, help='number of instances to delete.')
    parser.add_argument('--seed', type=int, default=423, help='seed to populate the data.')
    parser.add_argument('--max_depth', type=int, default=4, help='maximum depth of the tree.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    parser.add_argument('--no_refit', action='store_true', default=False, help='do not record refits.')
    args = parser.parse_args()
    print(args)
    main(args)
