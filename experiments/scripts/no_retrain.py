"""
Experiment: Find smallest epsilon/lmbda that gives a predictive performance
            within a tolerance % of the deterministic model.
"""
import os
import sys
import time
import argparse
from itertools import product

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import cedar
from utility import data_util, exp_util, print_util


def _get_model(args, lmbda, seed=None):
    """
    Return the appropriate model CeDAR model.
    """

    if args.model_type == 'stump':
        model = cedar.Tree(lmbda=lmbda,
                           max_depth=1,
                           verbose=args.verbose,
                           random_state=seed)

    elif args.model_type == 'tree':
        model = cedar.Tree(lmbda=lmbda,
                           max_depth=args.max_depth,
                           verbose=args.verbose,
                           random_state=seed)

    elif args.model_type == 'forest':
        model = cedar.Forest(lmbda=lmbda,
                             max_depth=args.max_depth,
                             n_estimators=args.n_estimators,
                             max_features=args.max_features,
                             verbose=args.verbose,
                             random_state=seed)

    else:
        exit('model_type {} unknown!'.format(args.model_type))

    return model


def experiment(args, logger, out_dir, seed):

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, seed, data_dir=args.data_dir)

    # dataset statistics
    logger.info('train instances: {:,}'.format(X_train.shape[0]))
    logger.info('test instances: {:,}'.format(X_test.shape[0]))
    logger.info('attributes: {:,}'.format(X_train.shape[1]))

    # tune on a fraction of the training data
    if args.tune_frac < 1.0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=2,
                                     train_size=args.tune_frac,
                                     random_state=seed)
        tune_indices, _ = list(sss.split(X_train, y_train))[0]
        X_train_sub, y_train_sub = X_train[tune_indices], y_train[tune_indices]
        logger.info('tune instances: {:,}'.format(X_train_sub.shape[0]))

    else:
        X_train_sub, y_train_sub = X_train, y_train

    # Deterministic performance
    logger.info('\nDeterministic')
    start = time.time()
    model = _get_model(args, lmbda=-1, seed=seed)
    exact_score = cross_val_score(model, X_train_sub, y_train_sub,
                                  scoring=args.scoring, cv=args.cv).mean()
    logger.info('[CV] {}: {:.3f}'.format(args.scoring, exact_score))

    model = model.fit(X_train, y_train)
    exp_util.performance(model, X_test, y_test, name='TEST', logger=logger)
    logger.info('total time: {:.3f}s'.format(time.time() - start))

    # find smallest lmbda that gives god performance
    logger.info('\nCeDAR')

    # hyperparameters
    if args.model_type == 'stump':
        param_grid = [{'max_depth': 1}]

    elif args.model_type == 'tree':
        max_depth_list = [1, 3, 5, 10, 20]
        max_depth_list = [x for x in max_depth_list if x <= args.max_depth]
        param_grid = [{'max_depth': max_depth} for max_depth in max_depth_list]

    elif args.model_type == 'forest':
        max_depths_list = [1, 3, 5, 10, 20]
        n_estimators_list = [10, 100, 1000]
        max_features_list = [0.25]

        max_depths_list = [x for x in max_depths_list if x <= args.max_depth]
        n_estimators_list = [x for x in n_estimators_list if x <= args.n_estimators]
        max_features_list = [x for x in max_features_list if x <= args.max_features]
        max_features_list.insert(0, 'sqrt')

        names = ['max_depth', 'n_estimators', 'max_features']
        values_list = list(product(max_depths_list, n_estimators_list, max_features_list))
        param_grid = [dict(zip(names, values)) for values in values_list]

    random_state = exp_util.get_random_state(seed)

    cedar_score = 0
    lmbda = -args.increment_lmbda
    finished = False

    while not finished:
        lmbda += args.increment_lmbda

        for params in param_grid:

            start = time.time()
            model = _get_model(args, lmbda=lmbda, seed=random_state)
            model.set_params(**params)
            cedar_score = cross_val_score(model, X_train, y_train,
                                          scoring=args.scoring, cv=args.cv).mean()

            end = time.time() - start
            out_str = '[{:.3f}s] params: {}, lmbda: {:.2e} => {}: {:.3f}'
            logger.info(out_str.format(end, params, lmbda, args.scoring, cedar_score))

            if exact_score - cedar_score <= args.tol:
                finished = True
                break

    model = model.fit(X_train, y_train)
    exp_util.performance(model, X_test, y_test, name='TEST', logger=logger)
    logger.info('lmbda: {}'.format(lmbda))

    n_removes = [1, 10, 100, 1000, int(0.005 * X_train.shape[0]), int(0.01 * X_train.shape[0])]
    epsilons = [lmbda * (n_remove / X_train.shape[0]) for n_remove in n_removes]

    logger.info('n_removes: {}'.format(n_removes))
    logger.info('epsilons: {}'.format(epsilons))

    if args.save_results:
        d = model.get_params()
        d['params'] = params
        d['lmbda'] = lmbda
        d['epsilon'] = epsilons
        d['n_remove'] = n_removes
        d['n_train'] = X_train.shape[0]
        d['n_features'] = X_train.shape[1]
        np.save(os.path.join(out_dir, 'results.npy'), d)


def main(args):

    # run experiment multiple times
    for i in range(args.repeats):

        # create output dir
        ep_dir = os.path.join(args.out_dir, args.dataset, args.model_type,
                              'rs{}'.format(args.rs))
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

    # experiment settings
    parser.add_argument('--out_dir', type=str, default='output/no_retrain/', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='surgical', help='dataset to use for the experiment.')
    parser.add_argument('--model_type', type=str, default='stump', help='stump, tree, or forest.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--repeats', type=int, default=1, help='number of times to perform the experiment.')
    parser.add_argument('--save_results', action='store_true', default=False, help='save results.')

    # tree/forest hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_features', type=float, default=None, help='maximum features to sample.')
    parser.add_argument('--max_depth', type=int, default=1, help='maximum depth of the tree.')
    parser.add_argument('--increment_lmbda', type=float, default=100, help='value to increment lmbda by.')

    parser.add_argument('--cv', type=int, default=2, help='Number of cross-validations.')
    parser.add_argument('--scoring', type=str, default='accuracy', help='Predictive performance metric.')
    parser.add_argument('--tune_frac', type=float, default=1.0, help='fraction of training to use for tuning.')
    parser.add_argument('--tol', type=float, default=0.01, help='Predictive performance tolerance.')

    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    main(args)
