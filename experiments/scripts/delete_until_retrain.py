"""
Experiment: How much data can we delete before the cumulative deletion time
            equals the original training time.
TODO: Record % deletions for exact unlearner.
"""
import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import cedar
from utility import data_util
from utility import exp_util
from utility import print_util
from utility import root_adversary


def _get_model(args, epsilon, lmbda, random_state=None):
    """
    Return the appropriate model CeDAR model.
    """

    if args.model_type == 'stump':
        model = cedar.Tree(epsilon=epsilon,
                           lmbda=lmbda,
                           max_depth=1,
                           criterion=args.criterion,
                           verbose=args.verbose,
                           random_state=random_state)

    elif args.model_type == 'tree':
        model = cedar.Tree(epsilon=epsilon,
                           lmbda=lmbda,
                           max_depth=args.max_depth,
                           criterion=args.criterion,
                           verbose=args.verbose,
                           random_state=random_state)

    elif args.model_type == 'forest':
        max_features = None if args.max_features == -1 else args.max_features
        model = cedar.Forest(epsilon=epsilon,
                             lmbda=lmbda,
                             max_depth=args.max_depth,
                             criterion=args.criterion,
                             n_estimators=args.n_estimators,
                             max_features=max_features,
                             verbose=args.verbose,
                             random_state=random_state)

    else:
        exit('model_type {} unknown!'.format(args.model_type))

    return model


def experiment(args, logger, out_dir, seed, lmbda):

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, data_dir=args.data_dir)

    # dataset statistics
    logger.info('train instances: {:,}'.format(X_train.shape[0]))
    logger.info('test instances: {:,}'.format(X_test.shape[0]))
    logger.info('features: {:,}'.format(X_train.shape[1]))
    logger.info('split criterion: {}'.format(args.criterion))

    # get random state
    random_state = exp_util.get_random_state(seed)

    # choose instances to delete
    n_remove = X_train.shape[0] - 1 if args.frac_remove is None else int(X_train.shape[0] * args.frac_remove)

    if args.adversary == 'random':
        np.random.seed(random_state)
        delete_indices = np.random.choice(X_train.shape[0], size=n_remove, replace=False)

    elif args.adversary == 'root':
        delete_indices = root_adversary.order_samples(X_train, y_train, n_samples=n_remove, seed=random_state,
                                                      verbose=args.verbose, logger=logger)
    else:
        exit('uknown adversary: {}'.format(args.adversary))

    logger.info('delete instances: {:,}'.format(len(delete_indices)))
    logger.info('adversary: {}'.format(args.adversary))

    # record time it takes to train a deterministic model
    logger.info('\nExact')
    model = _get_model(args, epsilon=0, lmbda=-1, random_state=random_state)
    start = time.time()
    model = model.fit(X_train, y_train)
    exact_train_time = time.time() - start

    logger.info('train time: {:.3f}s'.format(exact_train_time))
    exp_util.performance(model, X_test, y_test, name='TEST', logger=logger)

    # Exact
    if args.exact:
        logger.info('\nExact')
        logger.info('random_state: {}'.format(random_state))

        start = time.time()
        remaining_time = exact_train_time

        model = _get_model(args, epsilon=0, lmbda=-1, random_state=random_state)
        model = model.fit(X_train, y_train)
        exp_util.performance(model, X_test, y_test, name='TEST', logger=logger)

        # delete instances until train time is exceeded
        j = 0
        while remaining_time > 0 or j == len(X_train):
            start2 = time.time()
            model.delete(delete_indices[j])
            delete_time = time.time() - start2

            remaining_time -= delete_time
            j += 1

        if args.verbose > 0:
            delete_types, delete_depths = model.get_removal_statistics()
            logger.info('delete_types: {}'.format(delete_types))
            logger.info('delete_depths: {}'.format(delete_depths))

        end = time.time() - start
        logger.info('[{:.3f}s] num deletions: {:,}'.format(end, j))

    if args.save_results:
        d = model.get_params()
        d['train_time'] = exact_train_time
        d['n_deletions'] = j
        d['n_train'] = X_train.shape[0]
        d['n_features'] = X_train.shape[1]
        d['adversary'] = args.adversary
        np.save(os.path.join(out_dir, 'exact.npy'), d)

        logger.info('total time: {:.3f}s'.format(time.time() - start))

    # CeDAR
    if args.cedar:
        logger.info('\nCeDAR')
        start = time.time()

        epsilons = [0, 0.001, 0.01, 0.1, 1.0, 10.0]
        logger.info('epsilons: {}'.format(epsilons))
        logger.info('lmbda: {}'.format(lmbda))
        logger.info('random_state: {}'.format(random_state))

        # test different epsilons
        n_deletions = []
        for i, epsilon in enumerate(epsilons):
            remaining_time = exact_train_time
            epsilon_start = time.time()

            model = _get_model(args, epsilon=epsilon, lmbda=lmbda, random_state=random_state)
            model = model.fit(X_train, y_train)

            if i == 0:
                exp_util.performance(model, X_test, y_test, name='TEST', logger=logger)
                print()

            # delete instances until train time is exceeded
            j = 0
            while remaining_time > 0 or j == len(X_train):
                start2 = time.time()
                model.delete(delete_indices[j])
                delete_time = time.time() - start2

                remaining_time -= delete_time
                j += 1

            if args.verbose > 0:
                delete_types, delete_depths = model.get_removal_statistics()
                logger.info('delete_types: {}'.format(delete_types))
                logger.info('delete_depths: {}'.format(delete_depths))

            epsilon_end = time.time() - epsilon_start
            logger.info('[{:.3f}s] epsilon: {:5} => num deletions: {:,}'.format(epsilon_end, epsilon, j))
            n_deletions.append(j)

        if args.save_results:
            d = model.get_params()
            d['train_time'] = exact_train_time
            d['n_deletions'] = n_deletions
            d['epsilon'] = epsilons
            d['n_train'] = X_train.shape[0]
            d['n_features'] = X_train.shape[1]
            d['adversary'] = args.adversary
            np.save(os.path.join(out_dir, 'results.npy'), d)

        logger.info('total time: {:.3f}s'.format(time.time() - start))


def main(args):

    # create output dir
    ep_dir = os.path.join(args.out_dir, args.dataset, args.model_type,
                          args.criterion, args.adversary,
                          'rs{}'.format(args.rs))
    os.makedirs(ep_dir, exist_ok=True)

    # create logger
    logger = print_util.get_logger(os.path.join(ep_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())
    logger.info('\nSeed: {}'.format(args.rs))

    # run experiment
    experiment(args, logger, ep_dir, seed=args.rs, lmbda=args.lmbda)

    # remove logger
    print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment settings
    parser.add_argument('--out_dir', type=str, default='output/delete_until_retrain', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='surgical', help='dataset to use for the experiment.')
    parser.add_argument('--model_type', type=str, default='forest', help='stump, tree, or forest.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--save_results', action='store_true', default=True, help='save results.')

    # method settings
    parser.add_argument('--cedar', action='store_true', default=False, help='run CEDAR.')
    parser.add_argument('--exact', action='store_true', default=False, help='run exact.')

    # adversary settings
    parser.add_argument('--frac_remove', type=float, default=None, help='fraction of instances to delete.')
    parser.add_argument('--adversary', type=str, default='random', help='type of adversarial ordering.')

    # tree/forest hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_features', type=float, default=-1, help='maximum features to sample.')
    parser.add_argument('--max_depth', type=int, default=1, help='maximum depth of the tree.')
    parser.add_argument('--criterion', type=str, default='gini', help='splitting criterion.')
    parser.add_argument('--lmbda', type=float, default=0, help='noise hyperparameter')

    # display settings
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')

    args = parser.parse_args()
    main(args)
