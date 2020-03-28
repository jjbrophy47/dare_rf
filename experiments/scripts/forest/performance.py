"""
This experiment tests the accuracy of the decision trees.
"""
import os
import sys
import time
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../../')
sys.path.insert(0, here + '/../../')
import cedar
from utility import data_util, exp_util, print_util


def performance(args, logger, seed):

    n_estimators = [10, 50, 100, 250, 500, 1000]
    max_depths = [5, 10, 20, 35, 50]
    max_features = ['sqrt', 0.1, 0.2, 0.3]

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, seed, data_dir=args.data_dir)

    # dataset statistics
    logger.info('train instances: {}'.format(X_train.shape[0]))
    logger.info('test instances: {}'.format(X_test.shape[0]))
    logger.info('attributes: {}'.format(X_train.shape[1]))

    logger.info('building sk_rf...')
    start = time.time()
    sk_rf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                   max_features=args.max_features, max_samples=args.max_samples,
                                   verbose=args.verbose, random_state=seed, bootstrap=args.bootstrap)
    if args.tune:
        sk_param_grid = {'n_estimators': n_estimators, 'max_depth': max_depths, 'max_features': max_features,
                         'bootstrap': [True, False]}
        gs = GridSearchCV(sk_rf, sk_param_grid, scoring=args.scoring, cv=args.cv, verbose=args.verbose)
        gs = gs.fit(X_train, y_train)
        sk_rf = gs.best_estimator_
        logger.info('best_params: {}'.format(gs.best_params_))
    else:
        sk_rf = sk_rf.fit(X_train, y_train)
    logger.info('{:.3f}s'.format(time.time() - start))

    logger.info('building d_rf...')
    start = time.time()
    d_rf = cedar.Forest(lmbda=-1, n_estimators=args.n_estimators,
                        max_features=args.max_features, max_depth=args.max_depth,
                        verbose=args.verbose, random_state=seed)
    if args.tune:
        d_param_grid = {'n_estimators': n_estimators, 'max_depth': max_depths, 'max_features': max_features}
        gs = GridSearchCV(d_rf, d_param_grid, scoring=args.scoring, cv=args.cv, verbose=args.verbose)
        gs = gs.fit(X_train, y_train)
        d_rf = gs.best_estimator_
        logger.info('best_params: {}'.format(gs.best_params_))
    else:
        d_rf = d_rf.fit(X_train, y_train)
    logger.info('{:.3f}s'.format(time.time() - start))

    logger.info('building cedar_rf...')
    start = time.time()
    cedar_rf = cedar.Forest(lmbda=args.lmbda, n_estimators=args.n_estimators,
                            max_features=args.max_features, max_depth=args.max_depth,
                            verbose=args.verbose, random_state=seed)
    if args.tune:
        cedar_param_grid = {'n_estimators': n_estimators, 'max_depth': max_depths, 'max_features': max_features}
        gs = GridSearchCV(cedar_rf, cedar_param_grid, scoring=args.scoring, cv=args.cv,
                          verbose=args.verbose)
        gs = gs.fit(X_train, y_train)
        cedar_rf = gs.best_estimator_
        logger.info('best_params: {}'.format(gs.best_params_))
    else:
        cedar_rf = cedar_rf.fit(X_train, y_train)
    logger.info('{:.3f}s'.format(time.time() - start))

    # display performance
    exp_util.performance(sk_rf, X_test, y_test, name='sk_rf', logger=logger)
    exp_util.performance(d_rf, X_test, y_test, name='d_rf', logger=logger)
    exp_util.performance(cedar_rf, X_test, y_test, name='cedar_rf', logger=logger)


def main(args):

    # create output dir
    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    # create logger
    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)

    # run experiment
    performance(args, logger, seed=args.rs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/forest/performance', help='output directory.')
    parser.add_argument('--dataset', default='mfc19', help='dataset to use for the experiment.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--lmbda', type=float, default=100, help='amount of noise to add to the model.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_features', type=str, default='sqrt', help='maximum features to sample.')
    parser.add_argument('--max_samples', type=str, default=None, help='maximum samples to use.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth of the tree.')
    parser.add_argument('--bootstrap', action='store_true', default=False, help='use bootstrapping (sklearn).')
    parser.add_argument('--tune', action='store_true', default=False, help='tune models.')
    parser.add_argument('--cv', type=int, default=3, help='number of cross-validation folds for tuning.')
    parser.add_argument('--scoring', type=str, default='accuracy', help='metric for tuning.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    main(args)
