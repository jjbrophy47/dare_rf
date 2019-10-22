"""
This script tests the delete functionality for the BABC tree.
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
sys.path.insert(0, here + '/..')
from mulan.trees.babc_tree import BABC_Tree


def _fit_delete_refit(X, y, delete_ndx, max_depth=4):
    """
    This method first a tree, efficiently deletes the target instance,
    and refits a new tree without the target instance, and
    returns the times each of these events.
    """
    result = {}

    start = time.time()
    t1 = BABC_Tree(max_depth=max_depth).fit(X, y)
    result['fit'] = time.time() - start

    X_new = np.delete(X, delete_ndx, axis=0)
    y_new = np.delete(y, delete_ndx)

    start = time.time()
    result['delete_type'] = t1.delete(delete_ndx)
    result['delete'] = time.time() - start

    start = time.time()
    t2 = BABC_Tree(max_depth=max_depth).fit(X_new, y_new)
    result['refit'] = time.time() - start

    result['refit_to_delete_ratio'] = result['refit'] / result['delete']

    assert t1.equals(t2)
    return result


def _display_results(results, args, out_dir='output/bbac/synthetic'):
    """
    Plot the average time of deletion for each deletion type.
    """
    df = pd.DataFrame(results)
    f = plt.figure(figsize=(20, 4))
    ax0 = f.add_subplot(141)
    ax1 = f.add_subplot(142, sharey=ax0)
    ax2 = f.add_subplot(143, sharey=ax0)
    ax3 = f.add_subplot(144)
    df.boxplot(column='fit', by='delete_type', ax=ax0)
    df.boxplot(column='refit', by='delete_type', ax=ax1)
    df.boxplot(column='delete', by='delete_type', ax=ax2)
    df.boxplot(column='refit_to_delete_ratio', by='delete_type', ax=ax3)

    ax0.set_ylabel('seconds')
    f.suptitle('samples: {}, attributes: {}'.format(args.n_samples, args.n_attributes))

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 's{}_a{}.pdf'.format(args.n_samples, args.n_attributes)),
                bbox_inches='tight')


def main(args):

    np.random.seed(args.seed)
    X = np.random.randint(2, size=(args.n_samples, args.n_attributes))

    np.random.seed(args.seed)
    y = np.random.randint(2, size=args.n_samples)

    results = []
    for i in range(X.shape[0]):
        result = _fit_delete_refit(X, y, i)

        if args.verbose > 0:
            print('\nDeleting instance {}'.format(i))
            print('delete_type: {}'.format(result['delete_type']))
            print('fit: {:.5f}'.format(result['fit']))
            print('refit: {:.5f}'.format(result['refit']))
            print('delete: {:.5f}'.format(result['delete']))
            print('refit-to-delete ratio: {:.3f}'.format(result['refit_to_delete_ratio']))

        results.append(result)

    _display_results(results, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=10, help='number of samples to generate.')
    parser.add_argument('--n_attributes', type=int, default=4, help='number of attributes to generate.')
    parser.add_argument('--seed', type=int, default=423, help='seed to populate the data.')
    parser.add_argument('--max_depth', type=int, default=4, help='maximum depth of the tree.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    print(args)
    main(args)
