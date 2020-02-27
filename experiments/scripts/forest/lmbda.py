"""
Experiment: Show how the tst accuracy changes as epsilon and lmbda vary.
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
from model import cedar
from utility import data_util, exp_util, print_util


def vary_lmbda(args, logger, out_dir, seed):

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, seed, data_dir=args.data_dir)

    # dataset statistics
    logger.info('train instances: {}'.format(X_train.shape[0]))
    logger.info('test instances: {}'.format(X_test.shape[0]))
    logger.info('attributes: {}'.format(X_train.shape[1]))

    logger.info('building d_tree...')
    start = time.time()
    d_rf = cedar.RF(lmbda=100000,
                    n_estimators=args.n_estimators, max_features=args.max_features,
                    max_samples=args.max_samples, max_depth=args.max_depth,
                    verbose=args.verbose, random_state=seed)
    d_rf = d_rf.fit(X_train, y_train)
    logger.info('{:.3f}s'.format(time.time() - start))

    exp_util.performance(d_rf, X_test, y_test, name='d_rf')

    # save deterministic results
    if args.save_results:
        proba = d_rf.predict_proba(X_test)[:, 1]
        pred = d_rf.predict(X_test)
        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, pred)
        np.save(os.path.join(out_dir, 'd_auc.npy'), auc)
        np.save(os.path.join(out_dir, 'd_acc.npy'), acc)

    # observe change in test acuracy as lambda varies
    aucs, accs = [], []
    lmbdas = np.linspace(args.min_lmbda, args.max_lmbda, args.n_lmbda)
    logger.info('lmbdas ({}): {}'.format(len(lmbdas), lmbdas))
    for i, lmbda in enumerate(lmbdas):
        logger.info('[{}] lmbda: {:.2f}...'.format(i, lmbda))
        start = time.time()
        dt_rf = cedar.RF(lmbda=lmbda,
                         n_estimators=args.n_estimators, max_features=args.max_features,
                         max_samples=args.max_samples, max_depth=args.max_depth,
                         verbose=args.verbose, random_state=seed)
        dt_rf = dt_rf.fit(X_train, y_train)

        proba = dt_rf.predict_proba(X_test)[:, 1]
        pred = dt_rf.predict(X_test)
        aucs.append(roc_auc_score(y_test, proba))
        accs.append(accuracy_score(y_test, pred))

        if args.verbose > 0:
            exp_util.performance(dt_rf, X_test, y_test, name='dt_rf', logger=logger)

    if args.save_results:
        np.save(os.path.join(out_dir, 'auc.npy'), aucs)
        np.save(os.path.join(out_dir, 'acc.npy'), accs)
        np.save(os.path.join(out_dir, 'lmbda.npy'), lmbdas)


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
    parser.add_argument('--max_features', type=str, default='sqrt', help='maximum features to sample.')
    parser.add_argument('--max_samples', type=str, default=None, help='maximum samples to use.')
    parser.add_argument('--max_depth', type=int, default=4, help='maximum depth of the trees.')
    parser.add_argument('--repeats', type=int, default=1, help='number of times to perform the experiment.')
    parser.add_argument('--save_results', action='store_true', default=False, help='save results.')
    parser.add_argument('--min_lmbda', type=float, default=1, help='minimum lambda.')
    parser.add_argument('--max_lmbda', type=float, default=10000, help='maximum lambda.')
    parser.add_argument('--n_lmbda', type=int, default=20, help='number of data points.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    main(args)
