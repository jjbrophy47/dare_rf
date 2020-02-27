"""
This experiment chooses m instances to delete.
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
from model import cedar
from utility import data_util, exp_util, print_util, exact_adv_util, cert_adv_util


def _adjust_indices(indices_to_delete):
    """
    Return an adjusted array of indices, taking into account removing rach on sequentially.

    Example:
    indices_to_delete = [4, 1, 10, 0] => desired result = [4, 1, 8, 0]
    """
    assert len(np.unique(indices_to_delete)) == len(indices_to_delete)
    indices = indices_to_delete.copy()
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if indices[i] < indices[j]:
                indices[j] -= 1
    return indices


def remove_sample(args, logger, out_dir, seed):

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, seed, data_dir=args.data_dir)

    # dataset statistics
    logger.info('train instances: {}'.format(X_train.shape[0]))
    logger.info('test instances: {}'.format(X_test.shape[0]))
    logger.info('attributes: {}'.format(X_train.shape[1]))

    # choose instances to delete
    n_remove = args.n_remove if args.frac_remove is None else int(X_train.shape[0] * args.frac_remove)
    if args.adversary == 'exact':
        delete_indices = exact_adv_util.exact_adversary(X_train, y_train, n_samples=n_remove, seed=seed,
                                                        verbose=args.verbose, logger=logger)
    elif args.adversary == 'certified':
        delete_indices = cert_adv_util.certified_adversary(X_train, y_train, epsilon=args.epsilon, lmbda=args.lmbda,
                                                           gamma=args.gamma, n_samples=n_remove, seed=seed,
                                                           verbose=args.verbose, logger=logger)
    else:
        np.random.seed(seed)
        delete_indices = np.random.choice(X_train.shape[0], size=n_remove, replace=False)
    adjusted_indices = _adjust_indices(delete_indices)
    logger.info('instances to delete: {}'.format(len(delete_indices)))

    # deterministic model - training
    logger.info('\nd_rf')
    start = time.time()
    d_rf = cedar.RF(epsilon=args.epsilon, lmbda=10000, gamma=args.gamma,
                    n_estimators=args.n_estimators, max_features=args.max_features,
                    max_samples=args.max_samples, max_depth=args.max_depth,
                    verbose=args.verbose, random_state=seed)
    d_rf = d_rf.fit(X_train, y_train)
    end_time = time.time() - start
    logger.info('train time: {:.3f}s'.format(end_time))

    if not args.no_retrain:

        # deterministic model - deleting (i.e. retraining)
        tree_delete_times = [end_time]
        X_train_new, y_train_new = X_train.copy(), y_train.copy()
        for i, delete_ndx in enumerate(adjusted_indices):

            X_train_new = np.delete(X_train_new, delete_ndx, axis=0)
            y_train_new = np.delete(y_train_new, delete_ndx)

            start = time.time()
            d_rf = cedar.RF(epsilon=args.epsilon, lmbda=10000, gamma=args.gamma,
                            n_estimators=args.n_estimators, max_features=args.max_features,
                            max_samples=args.max_samples, max_depth=args.max_depth,
                            verbose=args.verbose, random_state=seed)
            d_rf = d_rf.fit(X_train, y_train)
            end_time = time.time() - start

            logger.info('{}. [{}] retrain time: {:.3f}s'.format(i, delete_ndx, end_time))
            tree_delete_times.append(end_time)

    else:
        tree_delete_times = [end_time] * (args.repeats + 1)

    # removal-enabled model - training
    logger.info('\ndt_rf')
    start = time.time()
    dt_rf = cedar.RF(epsilon=args.epsilon, lmbda=args.lmbda, gamma=args.gamma,
                     n_estimators=args.n_estimators, max_features=args.max_features,
                     max_samples=args.max_samples, max_depth=args.max_depth,
                     verbose=args.verbose, random_state=seed)
    dt_rf = dt_rf.fit(X_train, y_train)
    end_time = time.time() - start
    logger.info('train time: {:.3f}s'.format(end_time))

    # removal-enabled model - deleting
    detrace_delete_types = []
    detrace_delete_times = [end_time]
    for i, delete_ndx in enumerate(delete_indices):

        start = time.time()
        delete_types = dt_rf.delete(int(delete_ndx))
        end_time = time.time() - start
        detrace_delete_types += delete_types

        logger.info('{}. [{}] delete time: {:.3f}s'.format(i, delete_ndx, end_time))
        detrace_delete_times.append(end_time)

    counter = Counter(detrace_delete_types)
    logger.info('\n[dt_rf] delete types: {}\n'.format(counter))

    # amortized runtime
    logger.info('[{}] amortized: {:.3f}s'.format('d_rf', np.mean(tree_delete_times)))
    logger.info('[{}] amortized: {:.3f}s'.format('dt_rf', np.mean(detrace_delete_times)))

    # log the predictive performance
    logger.info('')
    exp_util.performance(d_rf, X_test, y_test, logger=logger, name='d_rf')
    exp_util.performance(dt_rf, X_test, y_test, logger=logger, name='dt_rf')
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
    parser.add_argument('--out_dir', type=str, default='output/forest/sequential_removal', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='synthetic', help='dataset to use for the experiment.')
    parser.add_argument('--rs', type=int, default=1, help='seed to enhance reproducibility.')
    parser.add_argument('--repeats', type=int, default=1, help='number of times to repeat the experiment.')
    parser.add_argument('--no_retrain', action='store_true', default=False, help='Do not retrain every time.')
    parser.add_argument('--n_remove', type=int, default=10, help='number of instances to sequentially delete.')
    parser.add_argument('--frac_remove', type=float, default=None, help='fraction of instances to delete.')
    parser.add_argument('--adversary', type=str, default=None, help='type of adversarial ordering.')
    parser.add_argument('--epsilon', type=float, default=0.1, help='idistinguishability parameter.')
    parser.add_argument('--lmbda', type=float, default=0.1, help='amount of noise to add to the model.')
    parser.add_argument('--gamma', type=float, default=0.1, help='fraction of data to support removal of.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_features', type=str, default='sqrt', help='maximum features to sample.')
    parser.add_argument('--max_samples', type=str, default=None, help='maximum samples to use.')
    parser.add_argument('--max_depth', type=int, default=4, help='maximum depth of the tree.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    main(args)
