"""
This experiment tests the accuracy of a single decision tree.
"""
import os
import sys
import time
import argparse

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../../')
sys.path.insert(0, here + '/../../')
import cedar
from utility import data_util, exp_util, print_util


def performance(args, logger, seed):

    param_grid = {'max_depth': [1, 3, 5, 10, 20]}
    logger.info('param_grid: {}'.format(param_grid))

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, seed, data_dir=args.data_dir)

    # dataset statistics
    logger.info('train instances: {:,}'.format(X_train.shape[0]))
    logger.info('test instances: {:,}'.format(X_test.shape[0]))
    logger.info('attributes: {:,}'.format(X_train.shape[1]))

    # SKLearn
    if args.sklearn:
        logger.info('\nSKLearn')
        start = time.time()
        model = DecisionTreeClassifier(max_depth=args.max_depth, random_state=seed)

        if args.tune:
            gs = GridSearchCV(model, param_grid, scoring=args.scoring, cv=args.cv, verbose=args.verbose)
            gs = gs.fit(X_train, y_train)
            model = gs.best_estimator_
            logger.info('best_params: {}'.format(gs.best_params_))
        else:
            model = model.fit(X_train, y_train)

        logger.info('{:.3f}s'.format(time.time() - start))
        exp_util.performance(model, X_test, y_test, name='sklearn', logger=logger)

    # Exact
    logger.info('\nExact')
    start = time.time()
    model = cedar.Tree(epsilon=args.epsilon, lmbda=-1, max_depth=args.max_depth,
                       verbose=args.verbose, random_state=seed)

    if args.tune:
        gs = GridSearchCV(model, param_grid, scoring=args.scoring, cv=args.cv, verbose=args.verbose)
        gs = gs.fit(X_train, y_train)
        model = gs.best_estimator_
        logger.info('best_params: {}'.format(gs.best_params_))
    else:
        model = model.fit(X_train, y_train)

    logger.info('{:.3f}s'.format(time.time() - start))
    exp_util.performance(model, X_test, y_test, name='exact', logger=logger)

    # CeDAR
    if args.cedar:
        logger.info('\nCeDAR')
        start = time.time()
        model = cedar.Tree(epsilon=args.epsilon, lmbda=args.lmbda,
                           max_depth=args.max_depth, verbose=args.verbose, random_state=seed)

        if args.tune:
            gs = GridSearchCV(model, param_grid, scoring=args.scoring, cv=args.cv, verbose=args.verbose)
            gs = gs.fit(X_train, y_train)
            model = gs.best_estimator_
            logger.info('best_params: {}'.format(gs.best_params_))
        else:
            model = model.fit(X_train, y_train)

        logger.info('{:.3f}s'.format(time.time() - start))
        exp_util.performance(model, X_test, y_test, name='cedar', logger=logger)


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
    parser.add_argument('--out_dir', type=str, default='output/tree/performance', help='output directory.')
    parser.add_argument('--dataset', default='mfc19', help='dataset to use for the experiment.')

    parser.add_argument('--rs', type=int, default=1, help='random state.')

    parser.add_argument('--tune', action='store_true', default=True, help='tune models.')
    parser.add_argument('--cv', type=int, default=2, help='number of cross-validation folds for tuning.')
    parser.add_argument('--scoring', type=str, default='accuracy', help='metric for tuning.')

    parser.add_argument('--sklearn', action='store_true', default=False, help='run sklearn model.')
    parser.add_argument('--cedar', action='store_true', default=False, help='run CeDAR model.')
    parser.add_argument('--epsilon', type=float, default=1.0, help='indistinguishability parameter.')
    parser.add_argument('--lmbda', type=float, default=100, help='amount of noise to add.')

    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth of the tree.')

    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    main(args)
