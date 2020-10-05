"""
Remove and Retrain (ROAR) experiment.
"""
import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import dart_rf as dart
from utility import data_util
from utility import exp_util
from utility import print_util

MAX_SEED_INCREASE = 1000


def _get_model(args):
    """
    Return model.
    """
    model = dart.Forest(criterion=args.criterion,
                        topd=0,
                        n_estimators=args.n_estimators,
                        max_features=args.max_features,
                        max_depth=args.max_depth,
                        random_state=args.rs)

    return model


def measure_performance(sort_indices, percentages, X_test, y_test, X_train, y_train,
                        logger=None):
    """
    Measures the change in log loss as training instances are removed.
    """
    r = {}
    aucs = []
    accs = []
    aps = []

    # remove training samples in batches
    for percentage in percentages:
        n_samples = int(X_train.shape[0] * (percentage / 100))
        remove_indices = sort_indices[:n_samples]

        new_X_train = np.delete(X_train, remove_indices, axis=0)
        new_y_train = np.delete(y_train, remove_indices)

        if len(np.unique(new_y_train)) == 1:
            print(percentage)
            break

        # train target model
        model = _get_model(args)
        label = '{}%'.format(percentage)
        model = model.fit(new_X_train, new_y_train)

        auc, acc, ap = exp_util.performance(model, X_test, y_test,
                                            logger=logger, name=label)
        aucs.append(auc)
        accs.append(acc)
        aps.append(ap)

    r['auc'] = aucs
    r['acc'] = accs
    r['ap'] = aps

    return r


def experiment(args, logger, out_dir):
    """
    Obtains data, trains model, and generates instance-attribution explanations.
    """

    # get data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, data_dir=args.data_dir)

    # select a subset of the test data for evaluation
    n_test_samples = args.n_test if args.n_test is not None else int(X_test.shape[0] * args.test_frac)
    np.random.seed(args.rs)
    test_indices = np.random.choice(X_test.shape[0], size=n_test_samples, replace=False)
    X_test_sub, y_test_sub = X_test[test_indices], y_test[test_indices]

    # choose new subset if test subset all contain the same label
    new_seed = args.rs
    while y_test_sub.sum() == len(y_test_sub) or y_test_sub.sum() == 0:
        np.random.seed(new_seed)
        new_seed += np.random.randint(MAX_SEED_INCREASE)
        np.random.seed(new_seed)
        test_indices = np.random.choice(X_test.shape[0], size=n_test_samples, replace=False)
        X_test_sub, y_test_sub = X_test[test_indices], y_test[test_indices]

    X_test = X_test_sub
    y_test = y_test_sub

    # dataset statistics
    logger.info('\ntrain instances: {:,}'.format(X_train.shape[0]))
    logger.info('test instances: {:,}'.format(X_test.shape[0]))
    logger.info('features: {:,}'.format(X_train.shape[1]))

    # experiment settings
    logger.info('\nrandom state: {}'.format(args.rs))
    logger.info('criterion: {}'.format(args.criterion))
    logger.info('n_estimators: {}'.format(args.n_estimators))
    logger.info('max_depth: {}'.format(args.max_depth))
    logger.info('max_features: {}'.format(args.max_features))
    logger.info('n_test: {}\n'.format(args.n_test))

    # train target model
    model = _get_model(args)
    name = 'D-DART'

    start = time.time()
    model = model.fit(X_train, y_train)
    train_time = time.time() - start

    logger.info('[{}] train time: {:.3f}s'.format(name, train_time))
    exp_util.performance(model, X_test, y_test, logger=logger, name=name)

    percentages = list(range(0, 100, 1))
    start = time.time()

    # random method
    if args.method == 'random':
        logger.info('\nordering by random...')
        np.random.seed(args.rs)
        train_order = np.random.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=False)
        results = measure_performance(train_order, percentages, X_test, y_test, X_train, y_train, logger)

    # D-DART 1: ordered from biggest sum increase in positive label confidence to least
    elif args.method == 'dart1':
        logger.info('\nordering by D-DART...')
        explanation = exp_util.explain_lite(model, X_train, y_train, X_test)
        train_order = np.argsort(explanation)[::-1]
        results = measure_performance(train_order, percentages, X_test, y_test, X_train, y_train, logger)

    # D-DART 2: ordered by most positively influential to least positively influential
    elif args.method == 'dart2':
        logger.info('\nordering by D-DART 2...')
        explanation = exp_util.explain_lite(model, X_train, y_train, X_test, y_test=y_test)
        train_order = np.argsort(explanation)[::-1]
        results = measure_performance(train_order, percentages, X_test, y_test, X_train, y_train, logger)

    # D-DART 3: ordered by biggest sum of absolute change in predictions
    elif args.method == 'dart3':
        logger.info('\nordering by D-DART 3...')
        explanation = exp_util.explain_lite(model, X_train, y_train, X_test, use_abs=True)
        train_order = np.argsort(explanation)[::-1]
        results = measure_performance(train_order, percentages, X_test, y_test, X_train, y_train, logger)

    logger.info('time: {:3f}s'.format(time.time() - start))

    results['percentage'] = percentages
    np.save(os.path.join(out_dir, 'results.npy'), results)


def main(args):

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.criterion,
                           args.method,
                           'rs_{}'.format(args.rs))

    log_fp = os.path.join(out_dir, 'log.txt')
    os.makedirs(out_dir, exist_ok=True)

    # skip experiment if results already exist
    if args.append_results and os.path.exists(os.path.join(out_dir, 'results.npy')):
        return

    # create logger
    logger = print_util.get_logger(log_fp)
    logger.info(args)
    logger.info(datetime.now())

    # run experiment
    experiment(args, logger, out_dir)

    # remove logger
    print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--out_dir', type=str, default='output/roar/', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='surgical', help='dataset to use for the experiment.')
    parser.add_argument('--append_results', action='store_true', default=False, help='add results.')

    # experiment settings
    parser.add_argument('--rs', type=int, default=1, help='seed to enhance reproducibility.')
    parser.add_argument('--n_test', type=int, default=50, help='no. test instances')
    parser.add_argument('--method', type=str, default='dart', help='method to use.')

    # tree hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_depth', type=int, default=10, help='maximum depth of the tree.')
    parser.add_argument('--max_features', type=float, default=0.25, help='maximum features to sample.')
    parser.add_argument('--criterion', type=str, default='gini', help='splitting criterion.')

    args = parser.parse_args()
    main(args)
