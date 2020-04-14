"""
This experiment tests the predictive performance of CeDAR.
"""
import os
import sys
import time
import argparse

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import cedar
from utility import data_util, exp_util, print_util


def _get_best_params(gs, param_grid, logger, tol=0.01):
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

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, data_dir=args.data_dir)

    # dataset statistics
    logger.info('train instances: {:,}'.format(X_train.shape[0]))
    logger.info('test instances: {:,}'.format(X_test.shape[0]))
    logger.info('attributes: {:,}'.format(X_train.shape[1]))

    # get random state
    random_state = exp_util.get_random_state(seed)

    # tune on a fraction of the training data
    if not args.no_tune:

        if args.tune_frac < 1.0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=2,
                                         train_size=args.tune_frac,
                                         random_state=random_state)
            tune_indices, _ = list(sss.split(X_train, y_train))[0]
            X_train_sub, y_train_sub = X_train[tune_indices], y_train[tune_indices]
            logger.info('tune instances: {:,}'.format(X_train_sub.shape[0]))

        else:
            X_train_sub, y_train_sub = X_train, y_train

    # hyperparameter values
    n_estimators = [10, 100, 1000]
    max_depth = [1, 3, 5, 10, 20]
    max_features = ['sqrt', 0.25]

    if args.model_type == 'tree':
        param_grid = {'max_depth': max_depth}

    elif args.model_type == 'forest':
        param_grid = {'max_depth': max_depth,
                      'n_estimators': n_estimators,
                      'max_features': max_features}

        # excludes the combination: 1000, 20, 0.25
        if args.reduce_search:
            param_grid = [{'max_depth': max_depth,
                           'n_estimators': [1, 100],
                           'max_features': max_features},
                          {'max_depth': [1, 3, 5, 10],
                           'n_estimators': [1000],
                           'max_features': max_features},
                          {'max_depth': [20],
                           'n_estimators': [1000],
                           'max_features': ['sqrt']}]

    # SKLearn
    if args.sklearn:
        logger.info('\nSKLearn')
        start = time.time()

        if args.model_type == 'stump':
            model = DecisionTreeClassifier(max_depth=1,
                                           random_state=random_state)

        elif args.model_type == 'tree':
            model = DecisionTreeClassifier(max_depth=args.max_depth,
                                           random_state=random_state)

        else:
            sk_max_features = 'sqrt' if not args.max_features else args.max_features
            model = RandomForestClassifier(max_depth=args.max_depth,
                                           n_estimators=args.n_estimators,
                                           max_features=sk_max_features)

        if not args.no_tune and args.model_type in ['tree', 'forest']:
            logger.info('param_grid: {}'.format(param_grid))
            gs = GridSearchCV(model, param_grid, scoring=args.scoring,
                              cv=args.cv, verbose=args.verbose)
            gs = gs.fit(X_train, y_train)
            model = gs.best_estimator_
            logger.info('best_params: {}'.format(gs.best_params_))

        else:
            model = model.fit(X_train, y_train)

        logger.info('{:.3f}s'.format(time.time() - start))
        exp_util.performance(model, X_test, y_test, name='sklearn', logger=logger)

    # Deterministic
    logger.info('\nDeterministic')
    start = time.time()

    if args.model_type == 'stump':
        model = cedar.Tree(lmbda=-1,
                           max_depth=1,
                           verbose=args.verbose,
                           random_state=random_state)

    elif args.model_type == 'tree':
        model = cedar.Tree(lmbda=-1,
                           max_depth=args.max_depth,
                           verbose=args.verbose,
                           random_state=random_state)

    elif args.model_type == 'forest':
        model = cedar.Forest(lmbda=-1,
                             max_depth=args.max_depth,
                             n_estimators=args.n_estimators,
                             max_features=args.max_features,
                             verbose=args.verbose,
                             random_state=random_state)

    if args.model_type in['tree', 'forest'] and not args.no_tune:

        logger.info('param_grid: {}'.format(param_grid))
        gs = GridSearchCV(model, param_grid, scoring=args.scoring,
                          cv=args.cv, verbose=args.verbose, refit=False)
        gs = gs.fit(X_train_sub, y_train_sub)

        best_params = _get_best_params(gs, param_grid, logger, args.tol)
        model.set_params(**best_params)

    model = model.fit(X_train, y_train)
    exp_util.performance(model, X_test, y_test, name='deterministic', logger=logger)
    logger.info('total time: {:.3f}s'.format(time.time() - start))


def main(args):

    # create output dir
    out_dir = os.path.join(args.out_dir, args.dataset, args.model_type)
    os.makedirs(out_dir, exist_ok=True)

    # create logger
    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)

    # run experiment
    performance(args, logger, seed=args.rs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment settings
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/performance/', help='output directory.')
    parser.add_argument('--dataset', default='mfc19', help='dataset to use for the experiment.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--model_type', type=str, default='stump', help='stump, tree, or forest.')

    # hyperparameter tuning settings
    parser.add_argument('--no_tune', action='store_true', default=False, help='do not tune.')
    parser.add_argument('--cv', type=int, default=2, help='number of cross-validation folds for tuning.')
    parser.add_argument('--scoring', type=str, default='roc_auc', help='metric for tuning.')
    parser.add_argument('--tune_frac', type=float, default=1.0, help='fraction of training to use for tuning.')
    parser.add_argument('--tol', type=float, default=0.01, help='allowable accuracy difference from the best.')
    parser.add_argument('--reduce_search', action='store_true', default=False, help='remove costly tuning.')

    # tree/forest hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_features', type=float, default=None, help='maximum features to sample.')
    parser.add_argument('--max_depth', type=int, default=1, help='maximum depth of the tree.')

    # sklearn specific hyperparameters
    parser.add_argument('--sklearn', action='store_true', default=False, help='run sklearn model.')
    parser.add_argument('--bootstrap', action='store_true', default=False, help='use bootstrapping (sklearn).')

    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    main(args)
