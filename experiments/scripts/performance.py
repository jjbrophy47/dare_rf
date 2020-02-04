"""
This experiment tests the accuracy of the decision trees.
BABC: Binary Attributes Binary Classification.
"""
import os
import sys
import time
import argparse

from sklearn.ensemble import RandomForestClassifier

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
from model import deterministic, detrace
from utility import data_util, exp_util


def main(args):

    # obtain data
    data = data_util.get_data(args.dataset, args.seed, data_dir=args.data_dir, n_samples=args.n_samples,
                              n_attributes=args.n_attributes, test_frac=args.test_frac, convert=False)
    X_train, X_test, y_train, y_test = data

    # dataset statistics
    print('train instances: {}'.format(X_train.shape[0]))
    print('test instances: {}'.format(X_test.shape[0]))
    print('attributes: {}'.format(X_train.shape[1]))

    print('building sk_rf...')
    start = time.time()
    sk_rf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                   max_features=args.max_features, max_samples=args.max_samples,
                                   verbose=args.verbose, random_state=args.seed)
    sk_rf = sk_rf.fit(X_train, y_train)
    print('{:.3f}s'.format(time.time() - start))

    print('building d_tree...')
    start = time.time()
    d_tree = deterministic.Tree(max_depth=args.max_depth, verbose=args.verbose)
    d_tree = d_tree.fit(X_train, y_train)
    print('{:.3f}s'.format(time.time() - start))

    print('building d_rf...')
    start = time.time()
    d_rf = deterministic.RF(n_estimators=args.n_estimators, max_features=args.max_features,
                            max_samples=args.max_samples, max_depth=args.max_depth, verbose=args.verbose,
                            random_state=args.seed)
    d_rf = d_rf.fit(X_train, y_train)
    print('{:.3f}s'.format(time.time() - start))

    print('building dtrace_rf...')
    start = time.time()
    dt_rf = detrace.RF(epsilon=args.epsilon, gamma=args.gamma, n_estimators=args.n_estimators,
                       max_features=args.max_features, max_samples=args.max_samples,
                       max_depth=args.max_depth, verbose=args.verbose, random_state=args.seed)
    dt_rf = dt_rf.fit(X_train, y_train)
    print('{:.3f}s'.format(time.time() - start))

    # display performance
    exp_util.performance(sk_rf, X_test, y_test, name='sk_rf')
    exp_util.performance(d_tree, X_test, y_test, name='d_tree')
    exp_util.performance(d_rf, X_test, y_test, name='d_rf')
    exp_util.performance(dt_rf, X_test, y_test, name='dt_rf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='synthetic', help='dataset to use for the experiment.')
    parser.add_argument('--n_samples', type=int, default=10, help='number of samples to generate.')
    parser.add_argument('--n_attributes', type=int, default=4, help='number of attributes to generate.')
    parser.add_argument('--test_frac', type=float, default=0.2, help='fraction of data to use for testing.')
    parser.add_argument('--seed', type=int, default=1, help='seed to populate the data.')
    parser.add_argument('--epsilon', type=float, default=0.1, help='efficiency parameter for tree.')
    parser.add_argument('--gamma', type=float, default=0.1, help='fraction of data to certifiably remove.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_features', type=str, default='sqrt', help='maximum features to sample.')
    parser.add_argument('--max_samples', type=str, default=None, help='maximum samples to use.')
    parser.add_argument('--max_depth', type=int, default=4, help='maximum depth of the tree.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    args = exp_util.check_args(args)
    print(args)
    main(args)
