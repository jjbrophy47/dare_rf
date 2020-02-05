"""
This experiment chooses m random instances to delete in batch.
TODO: compare against retraining a deterministic model or our model?
"""
import os
import sys
import time
import argparse
from collections import Counter

import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
from mulan.trees.babc_tree_d import BABC_Tree_D
from mulan.trees.tree import Tree
from utility import data_util, exp_util, print_util


def remove_sample(args, logger, out_dir, seed):

    # obtain data
    data = data_util.get_data(args.dataset, seed, data_dir=args.data_dir, n_samples=args.n_samples,
                              n_attributes=args.n_attributes, test_frac=args.test_frac, convert=False)
    X_train, X_test, y_train, y_test = data

    # dataset statistics
    logger.info('train instances: {}'.format(X_train.shape[0]))
    logger.info('test instances: {}'.format(X_test.shape[0]))
    logger.info('attributes: {}'.format(X_train.shape[1]))

    # choose instances to delete
    n_remove = int(X_train.shape[0] * args.frac_remove) if args.frac_remove is not None else args.n_remove
    np.random.seed(seed)
    delete_indices = np.random.choice(X_train.shape[0], size=n_remove, replace=False)
    logger.info('instances to delete: {}'.format(len(delete_indices)))

    # deterministic model - training
    logger.info('\nBABC_Tree_D')
    start = time.time()
    tree = BABC_Tree_D(max_depth=args.max_depth).fit(X_train, y_train)
    end_time = time.time() - start
    logger.info('train time: {:.3f}s'.format(end_time))

    # deterministic model - deleting (i.e. retraining)
    X_train_new = np.delete(X_train, delete_indices, axis=0)
    y_train_new = np.delete(y_train, delete_indices)
    start = time.time()
    tree = BABC_Tree_D(max_depth=args.max_depth).fit(X_train_new, y_train_new)
    end_time = time.time() - start
    logger.info('retrain time: {:.3f}s'.format(end_time))

    # removal-enabled model - training
    logger.info('\nDeTRACE')
    start = time.time()
    dtrace = Tree(epsilon=args.epsilon, gamma=args.gamma, max_depth=args.max_depth, verbose=args.verbose)
    dtrace = dtrace.fit(X_train, y_train)
    end_time = time.time() - start
    logger.info('train time: {:.3f}s'.format(end_time))

    # removal-enabled model - deleting
    start = time.time()
    detrace_delete_types = dtrace.delete(delete_indices)
    end_time = time.time() - start
    logger.info('delete time: {:.3f}s'.format(end_time))

    # log the deletion types
    counter = Counter(detrace_delete_types)
    logger.info('\n[DeTRACE] delete types: {}'.format(counter))

    # log the predictive performance
    logger.info('')
    exp_util.performance(tree, X_test, y_test, logger=logger, name='BABC_Tree_D')
    exp_util.performance(dtrace, X_test, y_test, logger=logger, name='DeTRACE')
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
    parser.add_argument('--out_dir', type=str, default='output/batch_removal', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='synthetic', help='dataset to use for the experiment.')
    parser.add_argument('--n_samples', type=int, default=10, help='number of samples to generate.')
    parser.add_argument('--n_attributes', type=int, default=4, help='number of attributes to generate.')
    parser.add_argument('--test_frac', type=float, default=0.2, help='fraction of data to use for testing.')
    parser.add_argument('--rs', type=int, default=1, help='seed to enhance reproducibility.')
    parser.add_argument('--repeats', type=int, default=1, help='number of times to repeat the experiment.')
    parser.add_argument('--n_remove', type=int, default=1, help='number of instances to sequentially delete.')
    parser.add_argument('--frac_remove', type=float, default=None, help='fraction of training data to remove.')
    parser.add_argument('--epsilon', type=float, default=0.1, help='efficiency parameter for tree.')
    parser.add_argument('--gamma', type=float, default=0.1, help='fraction of data to certifiably remove.')
    parser.add_argument('--max_depth', type=int, default=4, help='maximum depth of the tree.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    main(args)
