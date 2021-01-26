"""
This experiment tunes the `topd` hyperparameter of
the DaRE trees.
"""
import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import dare
from utility import data_util
from utility import print_util


def _get_model(args, topd=0):
    """
    Return model with the specified `topd`.
    """

    model = dare.Forest(max_depth=args.max_depth,
                        criterion=args.criterion,
                        topd=topd,
                        k=args.k,
                        n_estimators=args.n_estimators,
                        max_features=args.max_features,
                        verbose=args.verbose,
                        random_state=args.rs)

    return model


def performance(args, out_dir, logger):

    begin = time.time()

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, data_dir=args.data_dir)

    # dataset statistics
    logger.info('\nno. train instances: {:,}'.format(X_train.shape[0]))
    logger.info('no. test instances: {:,}'.format(X_test.shape[0]))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))
    logger.info('split criterion: {}'.format(args.criterion))
    logger.info('scoring: {}'.format(args.scoring))

    # tune on a fraction of the training data
    if args.tune_frac < 1.0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=2,
                                     train_size=args.tune_frac,
                                     random_state=args.rs)
        tune_indices, _ = list(sss.split(X_train, y_train))[0]
        X_train_sub, y_train_sub = X_train[tune_indices], y_train[tune_indices]
        logger.info('tune instances: {:,}'.format(X_train_sub.shape[0]))

    else:
        X_train_sub, y_train_sub = X_train, y_train

    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.rs)

    # train exact model
    start = time.time()
    model = _get_model(args, topd=0)
    exact_score = cross_val_score(model, X_train_sub, y_train_sub, scoring=args.scoring, cv=skf).mean()
    logger.info('\n[topd=0] CV score: {:.5f}, time: {:.3f}s'.format(exact_score, time.time() - start))

    # train topd=0 model
    s = '[topd={}] CV score: {:.5f}, CV diff: {:.5f}, time: {:.3f}s'
    scores = {}
    best_scores = {tol: 0 for tol in args.tol}

    for topd in range(1, args.max_depth + 1):
        start = time.time()

        # obtain score for this topd
        model = _get_model(args, topd=topd)
        score = cross_val_score(model, X_train_sub, y_train_sub, scoring=args.scoring, cv=skf).mean()
        score_diff = exact_score - score
        scores[topd] = score
        end = time.time() - start

        logger.info(s.format(topd, score, score_diff, end))

        # update best score for each tolerance
        for tol in args.tol:
            if best_scores[tol] == topd - 1 and score_diff <= tol:
                best_scores[tol] = topd

        total_time = time.time() - begin

    logger.info('{}, total time: {:.3f}s'.format(best_scores, total_time))
    np.save(os.path.join(out_dir, 'results.npy'), best_scores)


def main(args):

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.criterion,
                           'rs_{}'.format(args.rs))

    os.makedirs(out_dir, exist_ok=True)

    # create logger
    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('timestamp: {}'.format(datetime.now()))

    # run experiment
    performance(args, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/topd_tuning/', help='output directory.')
    parser.add_argument('--dataset', default='surgical', help='dataset to use for the experiment.')

    # experiment settings
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--cv', type=int, default=5, help='number of cross-validation folds for tuning.')
    parser.add_argument('--scoring', type=str, default='roc_auc', help='metric for tuning.')
    parser.add_argument('--tune_frac', type=float, default=1.0, help='fraction of training to use for tuning.')
    parser.add_argument('--tol', type=float, default=[0.001, 0.0025, 0.005, 0.01], help='allowable metric difference.')

    # tree/forest hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_features', type=str, default='sqrt', help='maximum features to sample.')
    parser.add_argument('--max_depth', type=int, default=1, help='maximum depth of the tree.')
    parser.add_argument('--k', type=int, default=10, help='no. thresholds to sample for greedy nodes.')
    parser.add_argument('--criterion', type=str, default='gini', help='splitting criterion.')

    # display settings
    parser.add_argument('--verbose', type=int, default=2, help='verbosity level.')

    args = parser.parse_args()
    main(args)
