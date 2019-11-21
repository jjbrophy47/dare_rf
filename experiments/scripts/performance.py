"""
This experiment tests the accuracy of the decision trees.
BABC: Binary Attributes Binary Classification.
"""
import os
import sys
import time
import argparse

import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
from mulan.trees.babc_tree_d import BABC_Tree_D
from mulan.trees.babc_tree_r import BABC_Tree_R
from utility import data_util, exp_util


def main(args):

    # obtain dataset
    if args.dataset == 'synthetic':

        np.random.seed(args.seed)
        X = np.random.randint(2, size=(args.n_samples, args.n_attributes))

        np.random.seed(args.seed)
        y = np.random.randint(2, size=args.n_samples)

        n_train = X.shape[0] - int(X.shape[0] * args.test_frac)
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

    else:
        X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, convert=False)

    n_samples = X_train.shape[0]
    n_attributes = X_train.shape[1]
    print('n_samples: {}, n_attributes: {}'.format(n_samples, n_attributes))

    print('building d tree...')
    start = time.time()
    td = BABC_Tree_D(max_depth=args.max_depth, verbose=args.verbose).fit(X_train, y_train)
    print('{:.3f}s'.format(time.time() - start))

    print('building r tree...')
    start = time.time()
    tr = BABC_Tree_R(max_depth=args.max_depth, verbose=args.verbose).fit(X_train, y_train)
    print('{:.3f}s'.format(time.time() - start))

    exp_util.performance(td, X_test, y_test, name='tree_D')
    exp_util.performance(tr, X_test, y_test, name='tree_R')

    td.print_tree()
    tr.print_tree()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='synthetic', help='dataset to use for the experiment.')
    parser.add_argument('--n_samples', type=int, default=10, help='number of samples to generate.')
    parser.add_argument('--n_attributes', type=int, default=4, help='number of attributes to generate.')
    parser.add_argument('--test_frac', type=float, default=0.2, help='fraction of data to use for testing.')
    parser.add_argument('--seed', type=int, default=423, help='seed to populate the data.')
    parser.add_argument('--max_depth', type=int, default=4, help='maximum depth of the tree.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    print(args)
    main(args)
