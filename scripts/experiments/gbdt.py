"""
This experiment tests the predictive performance of a GBDT model.
"""
import os
import sys
import time
import argparse
from datetime import datetime
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

from utility import data_util
from utility import exp_util
from utility import print_util


def performance(args, logger):

    begin = time.time()

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, data_dir=args.data_dir)

    # dataset statistics
    logger.info('train instances: {:,}'.format(X_train.shape[0]))
    logger.info('test instances: {:,}'.format(X_test.shape[0]))
    logger.info('attributes: {:,}'.format(X_train.shape[1]))

    # tune on a fraction of the training data
    if not args.no_tune:

        if args.tune_frac < 1.0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=2,
                                         train_size=args.tune_frac,
                                         random_state=args.rs)
            tune_indices, _ = list(sss.split(X_train, y_train))[0]
            X_train_sub, y_train_sub = X_train[tune_indices], y_train[tune_indices]
            logger.info('tune instances: {:,}'.format(X_train_sub.shape[0]))

        else:
            X_train_sub, y_train_sub = X_train, y_train

    # hyperparameter values
    n_estimators = [10, 50, 100, 250]
    max_depth = [1, 3, 5, 10, 20]

    param_grid = {'max_depth': max_depth,
                  'n_estimators': n_estimators}

    # test model
    logger.info('\n{}'.format(args.model_type.capitalize()))
    start = time.time()

    # get model
    model = lgb.LGBMClassifier(num_leaves=2**10)

    # tune model
    if args.no_tune:
        model = model.fit(X_train, y_train)

    else:
        logger.info('param_grid: {}'.format(param_grid))
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.rs)
        gs = GridSearchCV(model, param_grid, scoring=args.scoring,
                          cv=skf, verbose=args.verbose, refit=True)
        gs = gs.fit(X_train_sub, y_train_sub)
        model = gs.best_estimator_
        logger.info('best params: {}'.format(gs.best_params_))

    # test model
    start = time.time()
    model = model.fit(X_train, y_train)
    exp_util.performance(model, X_test, y_test, name=args.model_type, logger=logger)
    logger.info('train time: {:.3f}s'.format(time.time() - start))
    logger.info('total time: {:.3f}s'.format(time.time() - begin))


def main(args):

    # create output dir
    out_dir = os.path.join(args.out_dir, args.dataset, args.model_type)

    if args.no_tune:
        out_dir = os.path.join(out_dir,
                               'trees_{}'.format(args.n_estimators),
                               'depth_{}'.format(args.max_depth))

    os.makedirs(out_dir, exist_ok=True)

    # create logger
    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    # run experiment
    performance(args, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment settings
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/gbdt/', help='output directory.')
    parser.add_argument('--dataset', default='surgical', help='dataset to use for the experiment.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--model_type', type=str, default='lgb', help='lightgbm.')

    # hyperparameter tuning settings
    parser.add_argument('--no_tune', action='store_true', default=False, help='do not tune.')
    parser.add_argument('--cv', type=int, default=5, help='number of cross-validation folds for tuning.')
    parser.add_argument('--scoring', type=str, default='roc_auc', help='metric for tuning.')
    parser.add_argument('--tune_frac', type=float, default=1.0, help='fraction of training to use for tuning.')

    # tree/forest hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_features', type=float, default=0.25, help='maximum features to sample.')
    parser.add_argument('--max_depth', type=int, default=10, help='maximum depth of the tree.')

    # display settings
    parser.add_argument('--verbose', type=int, default=2, help='verbosity level.')

    args = parser.parse_args()
    main(args)
