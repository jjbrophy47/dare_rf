"""
Experiment: Find smallest epsilon/lmbda that gives a predictive performance
            within a tolerance % of the deterministic model.
"""
import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import cedar
from utility import data_util, exp_util, print_util


def _get_model(args, lmbda, random_state):
    """
    Return the appropriate model CeDAR model.
    """

    if args.model_type == 'stump':
        model = cedar.Tree(lmbda=lmbda,
                           max_depth=1,
                           verbose=args.verbose,
                           random_state=random_state)

    elif args.model_type == 'tree':
        model = cedar.Tree(lmbda=lmbda,
                           max_depth=args.max_depth,
                           verbose=args.verbose,
                           random_state=random_state)

    elif args.model_type == 'forest':
        model = cedar.Forest(lmbda=lmbda,
                             max_depth=args.max_depth,
                             n_estimators=args.n_estimators,
                             max_features=args.max_features,
                             verbose=args.verbose,
                             random_state=random_state)

    else:
        exit('model_type {} unknown!'.format(args.model_type))

    return model


def experiment(args, logger, out_dir, seed):

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, data_dir=args.data_dir)

    # dataset statistics
    logger.info('train instances: {:,}'.format(X_train.shape[0]))
    logger.info('test instances: {:,}'.format(X_test.shape[0]))
    logger.info('attributes: {:,}'.format(X_train.shape[1]))

    # get random state
    random_state = exp_util.get_random_state(seed)

    # tune on a fraction of the training data
    if args.tune_frac < 1.0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=2,
                                     train_size=args.tune_frac,
                                     random_state=random_state)
        tune_indices, _ = list(sss.split(X_train, y_train))[0]
        X_train_sub, y_train_sub = X_train[tune_indices], y_train[tune_indices]
        logger.info('tune instances: {:,}'.format(X_train_sub.shape[0]))

    else:
        X_train_sub, y_train_sub = X_train, y_train

    # Deterministic performance
    logger.info('\nDeterministic')
    start = time.time()
    model = _get_model(args, lmbda=-1, random_state=random_state)
    exact_score = cross_val_score(model, X_train_sub, y_train_sub,
                                  scoring=args.scoring, cv=args.cv).mean()
    logger.info('[CV] {}: {:.3f}'.format(args.scoring, exact_score))

    model = model.fit(X_train, y_train)
    exp_util.performance(model, X_test, y_test, name='TEST', logger=logger)
    logger.info('total time: {:.3f}s'.format(time.time() - start))

    # find smallest lmbda that gives good performance
    logger.info('\nCeDAR')
    logger.info('random_state: {}'.format(random_state))

    cedar_score = 0
    lmbda = -args.lmbda_step_size

    i = -1
    while True:
        lmbda += args.lmbda_step_size
        i += 1

        start = time.time()
        model = _get_model(args, lmbda=lmbda, random_state=random_state)
        cedar_score = cross_val_score(model, X_train_sub, y_train_sub,
                                      scoring=args.scoring, cv=args.cv).mean()

        end = time.time() - start
        out_str = '[{:.3f}s] lmbda: {:.2e} => {}: {:.3f}'
        logger.info(out_str.format(end, lmbda, args.scoring, cedar_score))

        if exact_score - cedar_score <= args.tol:
            break

    logger.info('lmbda: {}'.format(lmbda))
    model = _get_model(args, lmbda=lmbda, random_state=random_state)
    model = model.fit(X_train, y_train)
    exp_util.performance(model, X_test, y_test, name='TEST', logger=logger)

    if args.save_results:
        d = model.get_params()
        d['lmbda'] = lmbda
        d['n_train'] = X_train.shape[0]
        d['n_features'] = X_train.shape[1]
        d['lmbda_step_size'] = args.lmbda_step_size
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
        logger.info(datetime.now())
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
    parser.add_argument('--repeats', type=int, default=10, help='number of times to perform the experiment.')
    parser.add_argument('--save_results', action='store_true', default=True, help='save results.')

    # tree/forest hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_features', type=float, default=None, help='maximum features to sample.')
    parser.add_argument('--max_depth', type=int, default=1, help='maximum depth of the tree.')

    # tuning settings
    parser.add_argument('--lmbda_step_size', type=float, default=100, help='value to increment lmbda by.')
    parser.add_argument('--cv', type=int, default=2, help='Number of cross-validations.')
    parser.add_argument('--scoring', type=str, default='roc_auc', help='Predictive performance metric.')
    parser.add_argument('--tune_frac', type=float, default=1.0, help='fraction of training to use for tuning.')
    parser.add_argument('--tol', type=float, default=0.01, help='Predictive performance tolerance.')

    # display settings
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')

    args = parser.parse_args()
    main(args)
