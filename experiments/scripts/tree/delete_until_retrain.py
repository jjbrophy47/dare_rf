"""
Experiment: How much data can we delete before the cumulative deletion time
            equals the original training time.
"""
import os
import sys
import time
import argparse

import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../..')
sys.path.insert(0, here + '/../..')
import cedar
from utility import data_util, exp_util, print_util, exact_adv_util


def experiment(args, logger, out_dir, seed):

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, seed, data_dir=args.data_dir)

    # choose instances to delete
    n_remove = X_train.shape[0] - 1 if args.frac_remove is None else int(X_train.shape[0] * args.frac_remove)

    if args.adversary == 'random':
        np.random.seed(seed)
        delete_indices = np.random.choice(X_train.shape[0], size=n_remove, replace=False)

    elif args.adversary == 'root':
        delete_indices = exact_adv_util.exact_adversary(X_train, y_train, n_samples=n_remove, seed=seed,
                                                        verbose=args.verbose, logger=logger)
    else:
        exit('uknown adversary: {}'.format(args.adversary))

    # dataset statistics
    logger.info('num train instances: {:,}'.format(X_train.shape[0]))
    logger.info('num test instances: {:,}'.format(X_test.shape[0]))
    logger.info('num features: {:,}'.format(X_train.shape[1]))
    logger.info('num delete instances: {:,}'.format(len(delete_indices)))
    logger.info('adversary: {}'.format(args.adversary))

    # record time it takes to train an exact model
    logger.info('\nExact')
    model = cedar.Tree(epsilon=0, lmbda=args.lmbda,
                       max_depth=args.max_depth,
                       verbose=args.verbose,
                       random_state=seed)

    start = time.time()
    model = model.fit(X_train, y_train)
    exact_train_time = time.time() - start

    logger.info('train time: {:.3f}s'.format(exact_train_time))
    exp_util.performance(model, X_test, y_test, name='exact', logger=logger)

    # delete as many instances as possible before reaching cumulative train time
    logger.info('\nCeDAR')

    epsilons = [0, 0.001, 0.01, 0.1, 1.0, 10.0]
    logger.info('epsilons: {}'.format(epsilons))
    random_state = exp_util.get_random_state(seed)

    n_deletions = []
    for i, epsilon in enumerate(epsilons):
        remaining_time = exact_train_time

        model = cedar.Tree(epsilon=epsilon, lmbda=args.lmbda,
                           max_depth=args.max_depth,
                           verbose=args.verbose,
                           random_state=random_state)
        model = model.fit(X_train, y_train)
        model.print(show_nodes=True)

        j = 0
        while remaining_time > 0 or j == len(X_train):
            start = time.time()
            model.delete(delete_indices[j])
            delete_time = time.time() - start

            remaining_time -= delete_time
            j += 1

        delete_types, delete_depths = model.get_removal_statistics()
        logger.info('epsilon: {:5} => num deletions: {:,}'.format(epsilon, j))
        n_deletions.append(j)

    if args.save_results:
        d = model.get_params()
        d['n_deletions'] = n_deletions
        d['epsilon'] = epsilons
        d['lmbda'] = args.lmbda
        d['n_train'] = X_train.shape[0]
        d['n_features'] = X_train.shape[1]
        d['adversary'] = args.adversary
        np.save(os.path.join(out_dir, 'results.npy'), d)


def main(args):

    # run experiment multiple times
    for i in range(args.repeats):

        # create output dir
        ep_dir = os.path.join(args.out_dir, args.dataset, args.adversary, 'rs{}'.format(args.rs))
        os.makedirs(ep_dir, exist_ok=True)

        # create logger
        logger = print_util.get_logger(os.path.join(ep_dir, 'log.txt'))
        logger.info(args)
        logger.info('\nRun {}, seed: {}'.format(i + 1, args.rs))

        # run experiment
        experiment(args, logger, ep_dir, seed=args.rs)
        args.rs += 1

        # remove logger
        print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='output/tree/delete_until_retrain', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='synthetic', help='dataset to use for the experiment.')

    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--repeats', type=int, default=1, help='number of times to perform the experiment.')
    parser.add_argument('--save_results', action='store_true', default=False, help='save results.')

    parser.add_argument('--frac_remove', type=float, default=None, help='fraction of instances to delete.')
    parser.add_argument('--adversary', type=str, default='random', help='type of adversarial ordering.')

    parser.add_argument('--lmbda', type=float, default=0, help='lambda.')
    parser.add_argument('--max_depth', type=int, default=1, help='maximum depth of the tree.')

    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    main(args)
