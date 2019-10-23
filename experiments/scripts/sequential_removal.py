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

    print(delete_ndx)

    X_new = np.delete(X_new, delete_ndx, axis=0)
    y_new = np.delete(y_new, delete_ndx)

    start = time.time()
    result['delete_type'] = t1.delete(delete_ndx)
    result['delete'] = time.time() - start

    start = time.time()
    t2 = BABC_Tree(max_depth=max_depth).fit(X_new, y_new)
    result['refit'] = time.time() - start

    result['refit_to_delete_ratio'] = result['refit'] / result['delete']

    t1.print_tree()
    t2.print_tree()

    assert t1.equals(t2)
    return result, t1, X_new, y_new


def _display_results(results, args, out_dir='output/bbac/sequential_removal'):
    """
    Plot the average time of deletion for each deletion type.
    """
    df = pd.DataFrame(results)
    f = plt.figure(figsize=(15, 4))
    ax0 = f.add_subplot(131)
    ax1 = f.add_subplot(132, sharey=ax0)
    ax2 = f.add_subplot(133)
    df.boxplot(column='refit', by='delete_type', ax=ax0)
    df.boxplot(column='delete', by='delete_type', ax=ax1)
    df.boxplot(column='refit_to_delete_ratio', by='delete_type', ax=ax2)

    ax0.set_ylabel('seconds')
    title_str = 'dataset: {}, samples: {}, attributes: {}, remove_frac: {}\n'
    f.suptitle(title_str.format(args.dataset, args.n_samples, args.n_attributes, args.remove_frac))

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 's{}_a{}.pdf'.format(args.n_samples, args.n_attributes)),
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

    if args.dataset == 'synthetic':

        np.random.seed(args.seed)
        X = np.random.randint(2, size=(args.n_samples, args.n_attributes))

        np.random.seed(args.seed)
        y = np.random.randint(2, size=args.n_samples)

    else:

        X, y = data_util.get_data(args.dataset)

    # retrieve the indices to remove
    n_remove = int(args.remove_frac * X.shape[0])
    indices_to_delete = np.random.choice(np.arange(X.shape[0]), size=n_remove, replace=False)
    adjusted_indices_to_delete = _adjust_indices(indices_to_delete)

    # create new mutable varibles to hold the decreasing datasets
    X_new, y_new = X.copy(), y.copy()
    t1 = BABC_Tree(max_depth=args.max_depth, verbose=1).fit(X.copy(), y.copy())

    t1.print_tree()

    results = []
    for i, ndx in enumerate(adjusted_indices_to_delete):
        result, t1, X_new, y_new = _fit_delete_refit(t1, X_new, y_new, int(ndx), max_depth=args.max_depth)

        if args.verbose > 0:
            if int(0.1 * n_remove) != 0 and i % int(0.1 * n_remove) == 0:
                print('removed: {:.2f}%'.format(i / X.shape[0] * 100))

        if args.verbose > 1:
            print('\nDeleting instance {}'.format(ndx))
            print('delete_type: {}'.format(result['delete_type']))
            print('refit: {:.5f}'.format(result['refit']))
            print('delete: {:.5f}'.format(result['delete']))
            print('refit-to-delete ratio: {:.3f}'.format(result['refit_to_delete_ratio']))

        results.append(result)

    # make sure decremental tree is the same as a learned tree without the desired indices
    X_temp, y_temp = np.delete(X, indices_to_delete, axis=0), np.delete(y, indices_to_delete)
    desired_tree = BABC_Tree(max_depth=args.max_depth).fit(X_temp, y_temp)
    assert t1.equals(desired_tree)

    _display_results(results, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='synthetic', help='dataset to use for the experiment.')
    parser.add_argument('--n_samples', type=int, default=10, help='number of samples to generate.')
    parser.add_argument('--n_attributes', type=int, default=4, help='number of attributes to generate.')
    parser.add_argument('--remove_frac', type=float, default=0.1, help='fraction of instances to delete.')
    parser.add_argument('--seed', type=int, default=423, help='seed to populate the data.')
    parser.add_argument('--max_depth', type=int, default=4, help='maximum depth of the tree.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    print(args)
    main(args)
