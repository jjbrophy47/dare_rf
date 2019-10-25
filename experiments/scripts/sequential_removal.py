"""
This experiment chooses m random instances to delete,
then deletes them sequentially.
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


def _fit_delete_refit(t1, X_new, y_new, delete_ndx, max_depth=4):
    """
    This method first a tree, efficiently deletes the target instance,
    and refits a new tree without the target instance, and
    returns the times for each of these events.
    """
    result = {}

    # print(X_new, y_new)
    # print(X_new[delete_ndx], y_new[delete_ndx], delete_ndx)

    # print(delete_ndx)

    del X_new[delete_ndx]
    del y_new[delete_ndx]

    start = time.time()
    result['delete_type'] = t1.delete(delete_ndx)
    result['delete'] = time.time() - start

    start = time.time()
    t2 = BABC_Tree(max_depth=max_depth).fit(X_new, y_new)
    result['refit'] = time.time() - start

    result['refit_to_delete_ratio'] = result['refit'] / result['delete']

    # t1.print_tree()
    # t2.print_tree()

    assert t1.equals(t2)
    return result, t1, X_new, y_new


def _display_results(results, args, n_samples, n_attributes, n_remove, out_dir='output/bbac/sequential_removal'):
    """
    Plot the average time of deletion for each deletion type.
    """
    df = pd.DataFrame(results)
    f = plt.figure(figsize=(20, 4))
    ax0 = f.add_subplot(141)
    ax1 = f.add_subplot(142)
    ax2 = f.add_subplot(143, sharey=ax1)
    ax3 = f.add_subplot(144)
    sns.countplot(x='delete_type', data=df, ax=ax0)
    df.boxplot(column='refit', by='delete_type', ax=ax1)
    df.boxplot(column='delete', by='delete_type', ax=ax2)
    df.boxplot(column='refit_to_delete_ratio', by='delete_type', ax=ax3)

    ax0.set_ylabel('count')
    ax0.set_xlabel('delete_type')
    ax0.set_title('deletion occurrences')
    ax1.set_ylabel('seconds')
    ax3.set_ylabel('ratio')
    title_str = 'dataset: {}, samples: {}, attributes: {}, removed: {}\n'
    f.suptitle(title_str.format(args.dataset, n_samples, n_attributes, n_remove))

    out_dir = os.path.join(out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 's{}_a{}_r{}.pdf'.format(n_samples, n_attributes, n_remove)),
                bbox_inches='tight')


def _adjust_indices(indices_to_delete):
    """
    Return an adjusted array of indices, taking into account removing rach on sequentially.

    Example:
    indices_to_delete = [4, 1, 10, 0] => desired result = [4, 1, 8, 0]
    """
    assert not _contains_duplicates(indices_to_delete)
    indices = indices_to_delete.copy()
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if indices[i] < indices[j]:
                indices[j] -= 1
    return indices


def _contains_duplicates(x):
    return len(np.unique(x)) != len(x)


def main(args):

    # obtain dataset
    if args.dataset == 'synthetic':

        np.random.seed(args.seed)
        X = np.random.randint(2, size=(args.n_samples, args.n_attributes))

        np.random.seed(args.seed)
        y = np.random.randint(2, size=args.n_samples)

        X, y = data_util.convert_data(X, y)

    else:

        X, _, y, _ = data_util.get_data(args.dataset)

    # retrieve the indices to remove
    n_remove = int(args.remove_frac * len(X.keys())) if args.n_remove is None else args.n_remove
    n_samples = len(X.keys())
    n_attributes = len(X[next(iter(X))])
    indices_to_delete = np.random.choice(list(X.keys()), size=n_remove, replace=False)

    print('n_samples: {}, n_attributes: {}'.format(n_samples, n_attributes))
    print(indices_to_delete)

    # create new mutable varibles to hold the decreasing datasets
    X_new, y_new = X.copy(), y.copy()
    t1 = BABC_Tree(max_depth=args.max_depth).fit(X.copy(), y.copy())

    # t1.print_tree()

    # delete instances one at a time and measure the time
    results = []
    for i, ndx in enumerate(indices_to_delete):
        result, t1, X_new, y_new = _fit_delete_refit(t1, X_new, y_new, int(ndx), max_depth=args.max_depth)

        if args.verbose > 0:
            if int(0.1 * n_remove) != 0 and i % int(0.1 * n_remove) == 0:
                print('removed: '.format(i))

        if args.verbose > 1:
            print('\nDeleting instance {}, adjusted: {}'.format(ndx))
            print('delete_type: {}'.format(result['delete_type']))
            print('refit: {:.5f}'.format(result['refit']))
            print('delete: {:.5f}'.format(result['delete']))
            print('refit-to-delete ratio: {:.3f}'.format(result['refit_to_delete_ratio']))

        results.append(result)

    # make sure decremental tree is the same as a learned tree without the desired indices
    desired_tree = BABC_Tree(max_depth=args.max_depth).fit(X_new, y_new)
    assert t1.equals(desired_tree)

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
    args = parser.parse_args()
    print(args)
    main(args)