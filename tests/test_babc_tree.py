"""
This script tests the delete functionality for the BABC tree.
BABC: Binary Attributes Binary Classification.
"""
import os
import sys
import time
import argparse

import numpy as np

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

    assert t1.equals(t2)
    return result


def main(args):

    np.random.seed(args.seed)
    X = np.random.randint(2, size=(args.n_samples, args.n_attributes))

    np.random.seed(args.seed)
    y = np.random.randint(2, size=args.n_samples)

    for i in range(X.shape[0]):
        print('\nDeleting instance {}'.format(i))
        result = _fit_delete_refit(X, y, i)

        print('delete_type: {}'.format(result['delete_type']))
        print('fit: {:.5f}'.format(result['fit']))
        print('refit: {:.5f}'.format(result['refit']))
        print('delete: {:.5f}'.format(result['delete']))
        print('delete-to-refit ratio: {:.3f}'.format(result['refit'] / result['delete']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=10, help='number of samples to generate.')
    parser.add_argument('--n_attributes', type=int, default=4, help='number of attributes to generate.')
    parser.add_argument('--seed', type=int, default=423, help='seed to populate the data.')
    parser.add_argument('--max_depth', type=int, default=4, help='maximum depth of the tree.')
    args = parser.parse_args()
    print(args)
    main(args)
