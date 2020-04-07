"""
This experiment tests the accuracy of the forest.
TODO: Tune/train on a subet of the training data.
"""
import os
import sys
import time
import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../../')
sys.path.insert(0, here + '/../../')
import cedar
from utility import data_util, exp_util, print_util


def _get_best_params(gs, param_grid, logger, tol=0.0):
    """
    Chooses the set of hyperparameters whose `mean_fit_score` is within
    `tol` of the best `mean_fit_score` and has the lowest `mean_fit_time`.
    """
    cols = ['mean_fit_time', 'mean_test_score', 'rank_test_score']
    cols += ['param_{}'.format(param) for param in param_grid.keys()]

    df = pd.DataFrame(gs.cv_results_)
    logger.info('gridsearch results:')
    logger.info(df[cols].sort_values('rank_test_score'))

    # filter the parameters with the highest performances
    logger.info('tolerance: {}'.format(args.tol))
    df = df[df['mean_test_score'].max() - df['mean_test_score'] <= tol]

    best_df = df.sort_values('mean_fit_time').reset_index().loc[0]
    best_ndx = best_df['index']
    best_params = best_df['params']
    logger.info('best_index: {}, best_params: {}'.format(best_ndx, best_params))

    return best_params


def performance(args, logger, seed):

    # hyperparameters
    n_estimators = [10, 100, 1000]
    max_depth = [1, 3, 5, 10, 20]
    max_features = ['sqrt', 0.25]

    logger.info('\nHyperparameters')
    logger.info('n_estimators: {}'.format(n_estimators))
    logger.info('max_depth: {}'.format(max_depth))
    logger.info('max_features: {}'.format(max_features))

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, seed, data_dir=args.data_dir)

    # dataset statistics
    logger.info('\nData')
    logger.info('train instances: {:,}'.format(X_train.shape[0]))
    logger.info('test instances: {:,}'.format(X_test.shape[0]))
    logger.info('attributes: {:,}'.format(X_train.shape[1]))

    # SKLearn
    if args.sklearn:
        logger.info('\nSKLearn')

        start = time.time()
        mf = 'sqrt' if not args.max_features else args.max_features
        model = RandomForestClassifier(n_estimators=args.n_estimators,
                                       max_depth=args.max_depth,
                                       max_features=mf, verbose=args.verbose,
                                       random_state=seed, bootstrap=args.bootstrap)

        if not args.no_tune:
            param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth,
                          'max_features': max_features, 'bootstrap': [True, False]}
            gs = GridSearchCV(model, param_grid, scoring=args.scoring, cv=args.cv,
                              verbose=args.verbose, refit=False)
            gs = gs.fit(X_train, y_train)
            best_params = _get_best_params(gs, param_grid, logger, args.tol)
            model = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                           max_depth=best_params['max_depth'],
                                           max_features=best_params['max_features'],
                                           bootstrap=best_params['bootstrap'],
                                           verbose=args.verbose, random_state=seed)

        model = model.fit(X_train, y_train)
        logger.info('{:.3f}s'.format(time.time() - start))
        exp_util.performance(model, X_test, y_test, name='sklearn', logger=logger)

    # Exact
    logger.info('\nExact')
    start = time.time()
    model = cedar.Forest(lmbda=-1, n_estimators=args.n_estimators,
                         max_features=args.max_features, max_depth=args.max_depth,
                         verbose=args.verbose, random_state=seed)
    if not args.no_tune:
        param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth,
                      'max_features': max_features}
        gs = GridSearchCV(model, param_grid, scoring=args.scoring, cv=args.cv,
                          verbose=args.verbose, refit=False)
        gs = gs.fit(X_train, y_train)
        best_params = _get_best_params(gs, param_grid, logger, args.tol)
        model = cedar.Forest(lmbda=-1, n_estimators=best_params['n_estimators'],
                             max_features=best_params['max_features'],
                             max_depth=best_params['max_depth'],
                             verbose=args.verbose, random_state=seed)

    model = model.fit(X_train, y_train)
    logger.info('{:.3f}s'.format(time.time() - start))
    exp_util.performance(model, X_test, y_test, name='exact', logger=logger)

    # CeDAR
    if args.cedar:
        logger.info('\nCeDAR')

        start = time.time()
        model = cedar.Forest(lmbda=args.lmbda, n_estimators=args.n_estimators,
                             max_features=args.max_features, max_depth=args.max_depth,
                             verbose=args.verbose, random_state=seed)

        if not args.no_tune:
            param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth,
                          'max_features': max_features}
            gs = GridSearchCV(model, param_grid, scoring=args.scoring, cv=args.cv,
                              verbose=args.verbose, refit=False)
            gs = gs.fit(X_train, y_train)
            best_params = _get_best_params(gs, param_grid, logger, args.tol)
            model = cedar.Forest(lmbda=args.lmbda, n_estimators=best_params['n_estimators'],
                                 max_features=best_params['max_features'],
                                 max_depth=best_params['max_depth'],
                                 verbose=args.verbose, random_state=seed)

        model = model.fit(X_train, y_train)
        logger.info('{:.3f}s'.format(time.time() - start))
        exp_util.performance(model, X_test, y_test, name='model', logger=logger)


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

    parser.add_argument('--sklearn', action='store_true', default=False, help='compare to an SKLearn model.')
    parser.add_argument('--cedar', action='store_true', default=False, help='compare to a CeDAR model.')

    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_features', type=float, default=None, help='maximum features to sample.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth of the tree.')
    parser.add_argument('--bootstrap', action='store_true', default=False, help='use bootstrapping (sklearn).')

    parser.add_argument('--no_tune', action='store_true', default=True, help='tune models.')
    parser.add_argument('--cv', type=int, default=2, help='number of cross-validation folds for tuning.')
    parser.add_argument('--scoring', type=str, default='accuracy', help='metric for tuning.')
    parser.add_argument('--tol', type=float, default=0.01, help='allowable accuracy difference from the best.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    main(args)
