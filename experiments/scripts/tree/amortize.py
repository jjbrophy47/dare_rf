"""
Experiment: Analyzes the amortized runtime under different deletion orerings.
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
import cedar
from utility import data_util, exp_util, print_util, exact_adv_util, cert_adv_util


def _adjust_indices(indices_to_delete):
    """
    Return an adjusted array of indices, taking into account removing each one sequentially.

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
                                                           n_samples=n_remove, seed=seed,
                                                           verbose=args.verbose, logger=logger)
    else:
        np.random.seed(seed)
        delete_indices = np.random.choice(X_train.shape[0], size=n_remove, replace=False)

    adjusted_indices = _adjust_indices(delete_indices)
    logger.info('instances to delete: {}'.format(len(delete_indices)))
    logger.info('adversary: {}'.format(args.adversary))

    # naive retraining method
    if args.retrain:

        # deterministic model - training
        logger.info('\nDeterministic')
        start = time.time()
        tree = cedar.Tree(epsilon=0, lmbda=-1, max_depth=args.max_depth,
                          verbose=args.verbose, random_state=seed)
        tree = tree.fit(X_train, y_train)
        end_time = time.time() - start
        logger.info('[{}] train time: {:.3f}s'.format('deterministic', end_time))
        exp_util.performance(tree, X_test, y_test, logger=logger, name='deterministic')

        if not args.no_retrain:

            # deterministic model - deleting (i.e. retraining)
            delete_times = [end_time]
            X_train_new, y_train_new = X_train.copy(), y_train.copy()
            for i, delete_ndx in enumerate(adjusted_indices):

                X_train_new = np.delete(X_train_new, delete_ndx, axis=0)
                y_train_new = np.delete(y_train_new, delete_ndx)

                start = time.time()
                tree = cedar.Tree(epsilon=0, lmbda=10**8, max_depth=args.max_depth,
                                  verbose=args.verbose, random_state=seed)
                tree = tree.fit(X_train_new, y_train_new)
                end_time = time.time() - start
                delete_times.append(end_time)

                if args.verbose > 0 and i % 100 == 0:
                    logger.info('{}. [{}] retrain time: {:.3f}s'.format(i, delete_ndx, end_time))

        else:
            X_train_new = np.delete(X_train, delete_indices, axis=0)
            y_train_new = np.delete(y_train, delete_indices)
            tree = cedar.Tree(epsilon=0, lmbda=10**8, max_depth=args.max_depth,
                              verbose=args.verbose, random_state=seed)
            tree = tree.fit(X_train_new, y_train_new)
            delete_times = [end_time] * (n_remove + 1)

        logger.info('[deterministic] amortized: {:.3f}s'.format(np.mean(delete_times)))
        exp_util.performance(tree, X_test, y_test, logger=logger, name='deterministic')

        if args.save_results:
            np.save(os.path.join(out_dir, 'retrain_time.npy'), delete_times)

    # deterministic method
    if args.exact:

        # exact unlearning model - training
        logger.info('\nExact')
        start = time.time()
        tree = cedar.Tree(epsilon=0, lmbda=-1, max_depth=args.max_depth,
                          verbose=args.verbose, random_state=seed)
        tree = tree.fit(X_train, y_train)
        end_time = time.time() - start
        logger.info('[exact] train time: {:.3f}s'.format(end_time))
        exp_util.performance(tree, X_test, y_test, logger=logger, name='exact')

        # exact unlearning model - deleting
        delete_times = [end_time]
        for i, delete_ndx in enumerate(delete_indices):
            print(delete_ndx)

            start = time.time()
            tree.delete(int(delete_ndx))
            end_time = time.time() - start
            delete_times.append(end_time)

            if args.verbose > 0 and i % 100 == 0:
                logger.info('{}. [{}] delete time: {:.3f}s'.format(i, delete_ndx, end_time))

        delete_types, delete_depths = tree.get_removal_statistics()
        type_counter = Counter(delete_types)
        depth_counter = Counter(delete_depths)
        logger.info('[exact] amortized: {:.3f}s'.format(np.mean(delete_times)))
        logger.info('[exact] delete types: {}'.format(type_counter))
        logger.info('[exact] delete depths: {}'.format(depth_counter))
        exp_util.performance(tree, X_test, y_test, logger=logger, name='exact')

        if args.save_results:
            np.save(os.path.join(out_dir, 'exact_time.npy'), delete_times)
            np.save(os.path.join(out_dir, 'exact_type.npy'), delete_types)

    # CeDAR method
    # for epsilon in [0.1, 1.0, 10.0]:
    for epsilon in [10.0]:

        # removal-enabled model - training
        logger.info('\nCeDAR (ep={}, lmbda={})'.format(epsilon, args.lmbda))
        start = time.time()
        tree = cedar.Tree(epsilon=epsilon, lmbda=args.lmbda, max_depth=args.max_depth,
                          verbose=args.verbose, random_state=seed)
        tree = tree.fit(X_train, y_train)
        end_time = time.time() - start
        logger.info('[cedar] train time: {:.3f}s'.format(end_time))
        exp_util.performance(tree, X_test, y_test, logger=logger, name='cedar')

        # removal-enabled model - deleting
        delete_times = [end_time]
        for i, delete_ndx in enumerate(delete_indices):

            start = time.time()
            tree.delete(int(delete_ndx))
            end_time = time.time() - start
            delete_times.append(end_time)

            if args.verbose > 0 and i % 100 == 0:
                logger.info('{}. [{}] delete time: {:.3f}s'.format(i, delete_ndx, end_time))

        # performance
        delete_types, delete_depths = tree.get_removal_statistics()
        type_counter = Counter(delete_types)
        depth_counter = Counter(delete_depths)
        logger.info('[cedar] amortized: {:.3f}s'.format(np.mean(delete_times)))
        logger.info('[cedar] delete types: {}'.format(type_counter))
        logger.info('[cedar] delete depths: {}'.format(depth_counter))
        exp_util.performance(tree, X_test, y_test, logger=logger, name='cedar')

        if args.save_results:
            np.save(os.path.join(out_dir, 'cedar_ep{}_time.npy'.format(epsilon)), delete_times)
            np.save(os.path.join(out_dir, 'cedar_ep{}_type.npy'.format(epsilon)), delete_types)


def main(args):

    # run experiment multiple times
    for i in range(args.repeats):

        # create output dir
        rs_dir = os.path.join(args.out_dir, args.dataset, '{}'.format(args.adversary), 'rs{}'.format(args.rs))
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
    parser.add_argument('--out_dir', type=str, default='output/tree/amortize', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='synthetic', help='dataset to use for the experiment.')
    parser.add_argument('--rs', type=int, default=1, help='seed to enhance reproducibility.')
    parser.add_argument('--repeats', type=int, default=1, help='number of times to repeat the experiment.')
    parser.add_argument('--retrain', action='store_true', default=False, help='Include retrain baseline.')
    parser.add_argument('--no_retrain', action='store_true', default=False, help='Do not retrain every time.')
    parser.add_argument('--exact', action='store_true', default=False, help='Include deterministic baseline.')
    parser.add_argument('--n_remove', type=int, default=10, help='number of instances to sequentially delete.')
    parser.add_argument('--frac_remove', type=float, default=None, help='fraction of instances to delete.')
    parser.add_argument('--adversary', type=str, default='random', help='type of adversarial ordering.')
    parser.add_argument('--epsilon', type=float, default=0.1, help='setting for certified adversarial ordering.')
    parser.add_argument('--lmbda', type=float, default=0.1, help='amount of noise to add to the model.')
    parser.add_argument('--max_depth', type=int, default=4, help='maximum depth of the tree.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    parser.add_argument('--save_results', action='store_true', default=False, help='save results.')
    args = parser.parse_args()
    main(args)
