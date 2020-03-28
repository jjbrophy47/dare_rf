"""
Experiment: Keeping the forest hyperparameters fixed, show how the test
            prediction performance changes as epsilon and lmbda vary.
"""
import os
import sys
import time
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../..')
sys.path.insert(0, here + '/../..')
import cedar
from utility import data_util, exp_util, print_util

MAX_INT = 2147483647


def _get_random_state(seed):
    np.random.seed(seed)
    return np.random.randint(MAX_INT)


def vary_lmbda(args, logger, out_dir, seed):

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, seed, data_dir=args.data_dir)

    # dataset statistics
    logger.info('train instances: {}'.format(X_train.shape[0]))
    logger.info('test instances: {}'.format(X_test.shape[0]))
    logger.info('attributes: {}'.format(X_train.shape[1]))

    logger.info('\nExact')
    start = time.time()
    model = cedar.Forest(lmbda=-1, n_estimators=args.n_estimators,
                         max_features=args.max_features, max_depth=args.max_depth,
                         verbose=args.verbose, random_state=_get_random_state(seed))
    model = model.fit(X_train, y_train)
    logger.info('{:.3f}s'.format(time.time() - start))

    exp_util.performance(model, X_test, y_test, name='exact')

    # save exact results
    if args.save_results:
        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, pred)

        d = model.get_params()
        d['auc'] = auc
        d['acc'] = acc
        d['n_train'] = X_train.shape[0]
        d['n_features'] = X_train.shape[1]
        np.save(os.path.join(out_dir, 'exact.npy'), d)

    # observe change in test performance as lambda varies
    logger.info('\nCeDAR')
    lmbdas = np.linspace(args.min_lmbda, args.max_lmbda, args.n_lmbda)
    logger.info('lmbdas: {}'.format(len(lmbdas)))

    aucs, accs = [], []
    random_state = _get_random_state(seed)
    for i, lmbda in enumerate(lmbdas):
        logger.info('[{}] lmbda: {:.2f}...'.format(i, lmbda))
        start = time.time()
        model = cedar.Forest(lmbda=lmbda, n_estimators=args.n_estimators,
                             max_features=args.max_features, max_depth=args.max_depth,
                             verbose=args.verbose, random_state=random_state)
        model = model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        aucs.append(roc_auc_score(y_test, proba))
        accs.append(accuracy_score(y_test, pred))

        if args.verbose > 0:
            exp_util.performance(model, X_test, y_test, name='cedar', logger=logger)

    if args.save_results:
        d = model.get_params()
        d['auc'] = aucs
        d['acc'] = accs
        d['lmbda'] = lmbdas
        d['n_train'] = X_train.shape[0]
        d['n_features'] = X_train.shape[1]
        np.save(os.path.join(out_dir, 'cedar.npy'), d)


def main(args):

    # run experiment multiple times
    for i in range(args.repeats):

        # create output dir
        ep_dir = os.path.join(args.out_dir, args.dataset, 'rs{}'.format(args.rs))
        os.makedirs(ep_dir, exist_ok=True)

        # create logger
        logger = print_util.get_logger(os.path.join(ep_dir, 'log.txt'))
        logger.info(args)
        logger.info('\nRun {}, seed: {}'.format(i + 1, args.rs))

        # run experiment
        vary_lmbda(args, logger, ep_dir, seed=args.rs)
        args.rs += 1

        # remove logger
        print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='output/forest/lmbda', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='synthetic', help='dataset to use for the experiment.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_features', type=float, default=None, help='maximum features to sample.')
    parser.add_argument('--max_depth', type=int, default=20, help='maximum depth of the trees.')
    parser.add_argument('--repeats', type=int, default=1, help='number of times to perform the experiment.')
    parser.add_argument('--save_results', action='store_true', default=False, help='save results.')
    parser.add_argument('--min_lmbda', type=float, default=1, help='minimum lambda.')
    parser.add_argument('--max_lmbda', type=float, default=10000, help='maximum lambda.')
    parser.add_argument('--n_lmbda', type=int, default=20, help='number of data points.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    main(args)
