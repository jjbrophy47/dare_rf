"""
This experiment tests the accuracy of a single decision tree.
"""
import os
import sys
import time
import argparse

from sklearn.tree import DecisionTreeClassifier

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../../')
sys.path.insert(0, here + '/../../')
import cedar
from utility import data_util, exp_util, print_util


def performance(args, logger, seed):

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, seed, data_dir=args.data_dir)

    # dataset statistics
    logger.info('train instances: {}'.format(X_train.shape[0]))
    logger.info('test instances: {}'.format(X_test.shape[0]))
    logger.info('attributes: {}'.format(X_train.shape[1]))

    logger.info('building sk_tree...')
    start = time.time()
    sk_tree = DecisionTreeClassifier(max_depth=args.max_depth, random_state=seed)
    sk_tree = sk_tree.fit(X_train, y_train)
    logger.info('{:.3f}s'.format(time.time() - start))

    print(sk_tree.tree_.node_count)
    print(sk_tree.tree_.max_depth)

    logger.info('building d_tree...')
    start = time.time()
    d_tree = cedar.Tree(epsilon=args.epsilon, lmbda=1000000000, max_depth=args.max_depth,
                        verbose=args.verbose, random_state=seed)
    d_tree = d_tree.fit(X_train, y_train)
    logger.info('{:.3f}s'.format(time.time() - start))

    logger.info('building CeDAR tree...')
    start = time.time()
    dt_tree = cedar.Tree(epsilon=args.epsilon, lmbda=args.lmbda,
                         max_depth=args.max_depth, verbose=args.verbose, random_state=seed)
    dt_tree = dt_tree.fit(X_train, y_train)
    logger.info('{:.3f}s'.format(time.time() - start))

    dt_tree.print_tree()

    # display performance
    exp_util.performance(sk_tree, X_test, y_test, name='sk_tree', logger=logger)
    exp_util.performance(d_tree, X_test, y_test, name='d_tree', logger=logger)
    exp_util.performance(dt_tree, X_test, y_test, name='cedar_tree', logger=logger)


def main(args):

    # create output dir
    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    # create logger
    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)

    # run experiment
    performance(args, logger, seed=args.rs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/tree/performance', help='output directory.')
    parser.add_argument('--dataset', default='synthetic', help='dataset to use for the experiment.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--epsilon', type=float, default=0.1, help='idistinguishability parameter.')
    parser.add_argument('--lmbda', type=float, default=0.1, help='amount of noise to add.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth of the tree.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    args = exp_util.check_args(args)
    main(args)
