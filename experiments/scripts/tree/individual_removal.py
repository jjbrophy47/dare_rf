"""
This experiment chooses a random instance to delete.
TODO: compare against retraining a deterministic model or our model?
"""
import os
import sys
import time
import argparse
from collections import Counter

import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../..')
sys.path.insert(0, here + '/../..')
from model import deterministic, detrace
from utility import data_util, exp_util, print_util


def remove_sample(args, logger, out_dir, seed):

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, seed, data_dir=args.data_dir)

    # dataset statistics
    logger.info('train instances: {}'.format(X_train.shape[0]))
    logger.info('test instances: {}'.format(X_test.shape[0]))
    logger.info('attributes: {}'.format(X_train.shape[1]))

    # choose instance to delete
    np.random.seed(seed)
    delete_ndx = np.random.choice(X_train.shape[0])
    logger.info('instance to delete: {}'.format(delete_ndx))

    # deterministic model - training
    logger.info('\nd_tree')
    start = time.time()
    d_tree = deterministic.Tree(max_depth=args.max_depth, verbose=args.verbose)
    d_tree = d_tree.fit(X_train, y_train)
    logger.info('train time: {:.3f}s'.format(time.time() - start))

    # deterministic model - deleting (i.e. retraining)
    X_train_new = np.delete(X_train, delete_ndx, axis=0)
    y_train_new = np.delete(y_train, delete_ndx)
    start = time.time()
    dt_tree = detrace.Tree(epsilon=args.epsilon, gamma=args.gamma, max_depth=args.max_depth,
                           verbose=args.verbose, random_state=seed)
    dt_tree = dt_tree.fit(X_train_new, y_train_new)
    logger.info('retrain time: {:.3f}s'.format(time.time() - start))

    # removal-enabled model - training
    logger.info('\ndt_tree')
    start = time.time()
    dt_tree = detrace.Tree(epsilon=args.epsilon, gamma=args.gamma, max_depth=args.max_depth,
                           verbose=args.verbose, random_state=seed)
    dt_tree = dt_tree.fit(X_train, y_train)
    logger.info('train time: {:.3f}s'.format(time.time() - start))

    # removal-enabled model - deleting
    start = time.time()
    deletion_types = dt_tree.delete(delete_ndx)
    logger.info('delete time: {:.3f}s'.format(time.time() - start))

    # display deletion types
    counter = Counter(deletion_types)
    logger.info('delete types: {}\n'.format(counter))

    # log the predictive performance
    exp_util.performance(d_tree, X_test, y_test, logger=logger, name='d_tree')
    exp_util.performance(dt_tree, X_test, y_test, logger=logger, name='dt_tree')
    logger.info('')


def main(args):

    # run experiment multiple times
    for i in range(args.repeats):

        # create output dir
        rs_dir = os.path.join(args.out_dir, args.dataset, 'rs{}'.format(args.rs))
        os.makedirs(rs_dir, exist_ok=True)

        # create logger
        logger = print_util.get_logger(os.path.join(rs_dir, 'log.txt'.format(args.dataset)))
        logger.info(args)
        logger.info('\nRun {}, seed: {}'.format(i + 1, args.rs))

        # run experiment
        remove_sample(args, logger, rs_dir, seed=args.rs)
        args.rs += 1

        # remove logger
        print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='output/tree/individual_removal', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='synthetic', help='dataset to use for the experiment.')
    parser.add_argument('--rs', type=int, default=1, help='seed to enhance reproducibility.')
    parser.add_argument('--repeats', type=int, default=1, help='number of times to repeat the experiment.')
    parser.add_argument('--epsilon', type=float, default=0.1, help='efficiency parameter for tree.')
    parser.add_argument('--gamma', type=float, default=0.1, help='fraction of data to certifiably remove.')
    parser.add_argument('--max_depth', type=int, default=4, help='maximum depth of the tree.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    args = exp_util.check_args(args)
    main(args)
