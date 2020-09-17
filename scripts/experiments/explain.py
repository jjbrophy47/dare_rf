"""
Generate an instance-attribution explanation.
"""
import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import dart_rf as dart
from utility import data_util
from utility import exp_util
from utility import print_util


def _get_model(args):
    """
    Return model.
    """
    model = dart.Forest(max_depth=args.max_depth,
                        criterion=args.criterion,
                        topd=0,
                        n_estimators=args.n_estimators,
                        max_features=args.max_features,
                        random_state=args.rs)

    return model


def experiment(args, logger, out_dir, seed):
    """
    Obtains data, trains model, and generates instance-attribution explanations.
    """

    # get data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, data_dir=args.data_dir)

    # select a subset of the test instances to explain
    test_indices = np.random.choice(X_test.shape[0], size=args.n_test, replace=False)
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]

    # dataset statistics
    logger.info('\ntrain instances: {:,}'.format(X_train.shape[0]))
    logger.info('test instances: {:,}'.format(X_test.shape[0]))
    logger.info('features: {:,}'.format(X_train.shape[1]))

    # experiment settings
    logger.info('\nrandom state: {}'.format(seed))
    logger.info('criterion: {}'.format(args.criterion))
    logger.info('n_estimators: {}'.format(args.n_estimators))
    logger.info('max_depth: {}'.format(args.max_depth))
    logger.info('max_features: {}'.format(args.max_features))
    logger.info('n_test: {}'.format(args.n_test))

    # begin experiment
    begin = time.time()

    # train target model
    model = _get_model(args)
    name = 'D-DART'

    start = time.time()
    model = model.fit(X_train, y_train)
    train_time = time.time() - start

    logger.info('[{}] train time: {:.3f}s'.format(name, train_time))

    auc_model, acc_model, ap_model = exp_util.performance(model, X_test, y_test,
                                                          logger=logger, name=name)

    # generate instance-attribution explanation
    initial_proba = model.predict_proba(X_test)[:, 1]
    impact = np.zeros(shape=(X_train.shape[0], X_test.shape[0]))

    for i in tqdm(range(X_train.shape[0])):
        model.delete(i)
        proba = model.predict_proba(X_test)[:, 1]
        impact[i] = initial_proba - proba
        model.add(X_train[[i]], y_train[[i]])

    # save model results
    results = model.get_params()
    results['model'] = name
    results['label'] = y_test[test_indices]
    results['impact'] = impact

    logger.info('{}'.format(results))
    np.save(os.path.join(out_dir, 'results.npy'), results)


def main(args):

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.criterion,
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
    experiment(args, logger, out_dir, seed=args.rs)

    # remove logger
    print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--out_dir', type=str, default='output/explain/', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='surgical', help='dataset to use for the experiment.')
    parser.add_argument('--append_results', action='store_true', default=False, help='add results.')

    # experiment settings
    parser.add_argument('--rs', type=int, default=1, help='seed to enhance reproducibility.')
    parser.add_argument('--n_test', type=int, default=100, help='no. test instances')

    # tree hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_depth', type=int, default=10, help='maximum depth of the tree.')
    parser.add_argument('--max_features', type=float, default=0.25, help='maximum features to sample.')
    parser.add_argument('--criterion', type=str, default='gini', help='splitting criterion.')

    args = parser.parse_args()
    main(args)
