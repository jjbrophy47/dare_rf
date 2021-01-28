"""
This experiment tests predictive performance.
"""
import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import dare
from utility import data_util
from utility import exp_util
from utility import print_util


def _get_model(args):
    """
    Return the appropriate model.
    """

    if args.model in ['dare']:
        model = dare.Forest(criterion=args.criterion,
                            max_depth=args.max_depth,
                            n_estimators=args.n_estimators,
                            max_features=args.max_features,
                            topd=args.topd,
                            k = args.k,
                            verbose=args.verbose,
                            random_state=args.rs)

    elif args.model == 'extra_trees':
        model = ExtraTreesClassifier(n_estimators=args.n_estimators,
                                     max_depth=args.max_depth,
                                     max_features=args.max_features,
                                     criterion=args.criterion,
                                     random_state=args.rs)

    elif args.model == 'extra_trees_k1':
        model = ExtraTreesClassifier(n_estimators=args.n_estimators,
                                     max_depth=args.max_depth,
                                     max_features=1,
                                     criterion=args.criterion,
                                     random_state=args.rs)

    elif args.model == 'borat':
        pass

    elif args.model == 'sklearn':
        model = RandomForestClassifier(n_estimators=args.n_estimators,
                                       max_depth=args.max_depth,
                                       max_features=args.max_features,
                                       criterion=args.criterion,
                                       random_state=args.rs,
                                       bootstrap=args.bootstrap)
    else:
        raise ValueError('model {} unknown!'.format(args.model))

    return model


def _get_model_dict(args, params):
    """
    Return the appropriate model.
    """

    if args.model == 'dare':
        model = dare.Forest(criterion=args.criterion,
                            max_depth=params['max_depth'],
                            n_estimators=params['n_estimators'],
                            max_features=args.max_features,
                            topd=args.topd,
                            k = params['k'],
                            verbose=args.verbose,
                            random_state=args.rs)

    elif args.model == 'extra_trees':
        model = ExtraTreesClassifier(n_estimators=params['n_estimators'],
                                     max_depth=params['max_depth'],
                                     max_features=args.max_features,
                                     criterion=args.criterion,
                                     random_state=args.rs)

    elif args.model == 'extra_trees_k1':
        model = ExtraTreesClassifier(n_estimators=params['n_estimators'],
                                     max_depth=params['max_depth'],
                                     max_features=1,
                                     criterion=args.criterion,
                                     random_state=args.rs)

    elif args.model == 'borat':
        pass

    elif args.model == 'sklearn':
        model = RandomForestClassifier(n_estimators=params['n_estimators'],
                                       max_depth=params['max_depth'],
                                       max_features=args.max_features,
                                       criterion=args.criterion,
                                       random_state=args.rs,
                                       bootstrap=args.bootstrap)
    else:
        raise ValueError('model {} unknown!'.format(args.model))

    return model


def _get_best_params(gs, param_grid, keys, logger, tol=1e-3):
    """
    Chooses the set of hyperparameters whose `mean_fit_score` is within
    `tol` of the best `mean_fit_score` and has the lowest `mean_fit_time`.
    """
    pd.set_option('display.max_columns', 100)

    cols = ['mean_fit_time', 'mean_test_score', 'rank_test_score']
    cols += ['param_{}'.format(param) for param in keys]

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


def performance(args, out_dir, logger):

    begin = time.time()

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, data_dir=args.data_dir)

    # dataset statistics
    logger.info('train instances: {:,}'.format(X_train.shape[0]))
    logger.info('test instances: {:,}'.format(X_test.shape[0]))
    logger.info('attributes: {:,}'.format(X_train.shape[1]))
    logger.info('split criterion: {}'.format(args.criterion))

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
    else:
        X_train_sub, y_train_sub = X_train, y_train

    # hyperparameter values
    n_estimators = [10, 50, 100, 250]
    max_depth = [1, 3, 5, 10, 20]

    # set hyperparameter grid
    param_grid = {'max_depth': max_depth,
                  'n_estimators': n_estimators}

    # add additional parameter for DaRE
    if args.model == 'dare':
        param_grid['k'] = [5, 10, 25, 50]

    # get hyperparameter names
    keys = list(param_grid.keys())

    # test model
    logger.info('\n{}'.format(args.model.capitalize()))
    start = time.time()
    model = _get_model(args)

    # if args.no_tune:
    #     train_indices = np.load('train_indices.npy')
    #     X_train_sub_temp, y_train_sub_temp = X_train_sub[train_indices], y_train_sub[train_indices]
    #     print(X_train_sub_temp.shape)
    #     model = _get_model(args)
    #     model.fit(X_train_sub_temp, y_train_sub_temp)
    # exit(0)

    # TEMPORARY
    # if args.no_tune:
    #     skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.rs)
    #     i = 0
    #     for train_indices, test_indices in skf.split(X_train_sub, y_train_sub):
    #         print(i)
    #         start = time.time()
    #         X_train_sub_temp, y_train_sub_temp = X_train_sub[train_indices], y_train_sub[train_indices]
    #         X_test_sub_temp, y_test_sub_temp = X_train_sub[test_indices], y_train_sub[test_indices]
    #         model = _get_model(args)
    #         model.fit(X_train_sub_temp, y_train_sub_temp)
    #         print('{}: {:.3f}s'.format(i, time.time() - start))
    #         np.save(os.path.join(out_dir, 'train_indices.npy'), train_indices)
    #         i += 1
    #     exit(0)

    # tune hyperparameters
    if not args.no_tune:
        logger.info('param_grid: {}'.format(param_grid))

        # cross-validation
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.rs)
        gs = GridSearchCV(model, param_grid, scoring=args.scoring,
                          cv=skf, verbose=args.verbose, refit=False)
        gs = gs.fit(X_train_sub, y_train_sub)

        best_params = _get_best_params(gs, param_grid, keys, logger, args.tol)
        model = _get_model_dict(args, best_params)

    # record time it takes to tune the model
    tune_time = time.time() - start

    # train best model
    start = time.time()
    model = model.fit(X_train, y_train)
    train_time = time.time() - start
    logger.info('train time: {:.3f}s'.format(train_time))

    # evaluate
    auc, acc, ap = exp_util.performance(model, X_test, y_test, name=args.model, logger=logger)

    # save results
    results = model.get_params()
    results['model'] = args.model
    results['bootstrap'] = args.bootstrap
    results['auc'] = auc
    results['acc'] = acc
    results['ap'] = ap
    results['train_time'] = train_time
    results['tune_train_time'] = tune_time + train_time
    np.save(os.path.join(out_dir, 'results.npy'), results)

    logger.info('total time: {:.3f}s'.format(time.time() - begin))


def main(args):

    # create output dir
    out_dir = os.path.join(args.out_dir, args.dataset, args.criterion)

    # add tuning to filepath
    if args.no_tune:
        out_dir = os.path.join(out_dir, 'no_tune', 'rs_{}'.format(args.rs))
    else:
        out_dir = os.path.join(out_dir, 'tuned', 'rs_{}'.format(args.rs))

    # create filename
    if args.model == 'sklearn':
        out_dir = os.path.join(out_dir, args.model)

        if args.bootstrap:
            out_dir = os.path.join(out_dir, 'bootstrap')

    elif args.model == 'dare':
        assert args.topd == 0
        out_dir = os.path.join(out_dir, args.model)

    elif args.model in ['extra_trees', 'extra_trees_k1', 'borat']:
        out_dir = os.path.join(out_dir, args.model)

    else:
        raise ValueError('model {} unknown!'.format(args.model))

    # create output directory and clear any previous contents
    os.makedirs(out_dir, exist_ok=True)
    print_util.clear_dir(out_dir)

    # create logger
    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    # write everything printed to stdout to this log file
    logfile, stdout, stderr = print_util.stdout_stderr_to_log(os.path.join(out_dir, 'log+.txt'))

    # run experiment
    performance(args, out_dir, logger)

    # restore original stdout and stderr settings
    print_util.reset_stdout_stderr(logfile, stdout, stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/performance/', help='output directory.')
    parser.add_argument('--dataset', default='surgical', help='dataset to use for the experiment.')

    # experiment settings
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--model', type=str, default='dare', help='type of model.')
    parser.add_argument('--criterion', type=str, default='gini', help='splitting criterion.')
    parser.add_argument('--topd', type=int, default=0, help='0 for exact, 1000 for random.')
    parser.add_argument('--k', type=int, default=10, help='no. of candidate thresholds to sample.')
    parser.add_argument('--bootstrap', action='store_true', default=False, help='use bootstrapping with sklearn.')

    # tuning settings
    parser.add_argument('--no_tune', action='store_true', default=False, help='do not tune.')
    parser.add_argument('--tune_frac', type=float, default=1.0, help='fraction of training to use for tuning.')
    parser.add_argument('--cv', type=int, default=5, help='number of cross-validation folds for tuning.')
    parser.add_argument('--scoring', type=str, default='roc_auc', help='metric for tuning.')
    parser.add_argument('--tol', type=float, default=1e-3, help='allowable accuracy difference from the best.')

    # tree/forest hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_features', type=str, default='sqrt', help='maximum no. features to sample.')
    parser.add_argument('--max_depth', type=int, default=10, help='maximum depth of the tree.')

    # display settings
    parser.add_argument('--verbose', type=int, default=2, help='verbosity level.')

    args = parser.parse_args()
    main(args)
