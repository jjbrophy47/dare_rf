"""
This experiment compares the memory usage of DARE models
against SKLearn's RandomForestClassifier implementation.
"""
import os
import sys
import time
import pickle
import argparse
import resource
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import dare
from utility import data_util
from utility import exp_util
from utility import print_util


def get_model(args, model, n_estimators, max_depth, topd=0, k=5):
    """
    Return the appropriate model.
    """

    if 'dare' in args.model:
        model = dare.Forest(criterion=args.criterion,
                            max_depth=max_depth,
                            n_estimators=n_estimators,
                            max_features=args.max_features,
                            topd=topd,
                            k=k,
                            verbose=args.verbose,
                            random_state=args.rs)

    elif 'sklearn' in model:
        bootstrap = True if 'bootstrap' in model else False
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       max_features=args.max_features,
                                       criterion=args.criterion,
                                       random_state=args.rs,
                                       bootstrap=bootstrap)
    else:
        raise ValueError('model {} unknown!'.format(args.model))

    return model


def experiment(args, out_dir, logger):

    # stat timer
    begin = time.time()

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, data_dir=args.data_dir)

    # compute data size
    data_mem = X_train.nbytes + y_train.nbytes

    # dataset statistics
    logger.info('\ntrain instances: {:,}'.format(X_train.shape[0]))
    logger.info('test instances: {:,}'.format(X_test.shape[0]))
    logger.info('attributes: {:,}'.format(X_train.shape[1]))
    logger.info('data size: {:,} bytes'.format(data_mem))

    # get hyperparameters
    params = exp_util.get_params(dataset=args.dataset, criterion=args.criterion)
    n_estimators = params[0]
    max_depth = params[1]
    k = params[2]
    topd_list = params[3:]
    tol_list = [0.0, 0.1, 0.25, 0.5, 1.0]
    assert len(topd_list) == len(tol_list)

    # create result object
    result = {}
    result['n_estimators'] = n_estimators
    result['max_depth'] = max_depth
    result['criterion'] = args.criterion
    result['max_features'] = args.max_features
    result['model'] = args.model

    # SKLearn RF
    if 'sklearn' in args.model:
        clf = get_model(args, model='sklearn', n_estimators=n_estimators, max_depth=max_depth)

        # train
        start = time.time()
        model = clf.fit(X_train, y_train)
        train_time = time.time() - start

        # get memory usage
        structure_memory = sys.getsizeof(pickle.dumps(model))
        decision_stats_memory = -1
        leaf_stats_memory = -1
        logger.info('\n[SKLearn] train: {:.3f}s, structure: {:,} bytes'.format(train_time, structure_memory))

    # DARE model
    else:

        # extract topd info
        dare_ndx = int(args.model.split('_')[1])
        tol = tol_list[dare_ndx]
        topd = topd_list[dare_ndx]

        # get model
        clf = get_model(args, model='dare', n_estimators=n_estimators, max_depth=max_depth, k=k, topd=topd)

        # train
        start = time.time()
        model = clf.fit(X_train, y_train)
        train_time = time.time() - start

        # get memory usage
        structure_memory, decision_stats_memory, leaf_stats_memory = model.get_memory_usage()
        s = '\n[DARE (tol={:.2f}%, topd={:,}, k={:,})] train: {:.3f}s'
        s += ', structure: {:,} bytes, decision stats.: {:,} bytes, leaf stats.: {:,} bytes'
        logger.info(s.format(tol, topd, k, train_time, structure_memory, decision_stats_memory, leaf_stats_memory))

    # add to results
    result['data_mem'] = data_mem
    result['structure_mem'] = structure_memory
    result['decision_stats_mem'] = decision_stats_memory
    result['leaf_stats_mem'] = leaf_stats_memory
    result['train_time'] = train_time

    # save results
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    np.save(os.path.join(out_dir, 'results.npy'), result)

    logger.info('\ntotal time: {:.3f}s'.format(time.time() - begin))
    logger.info('max_rss: {:,}'.format(result['max_rss']))
    logger.info('\nresults:\n{}'.format(result))


def main(args):

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.criterion,
                           args.model,
                           'rs_{}'.format(args.rs))

    # create output directory and clear any previous contents
    os.makedirs(out_dir, exist_ok=True)
    print_util.clear_dir(out_dir)

    # create logger
    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('\ntimestamp: {}'.format(datetime.now()))

    # run experiment
    experiment(args, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/memory/', help='output directory.')
    parser.add_argument('--dataset', default='surgical', help='dataset to use for the experiment.')

    # experiment settings
    parser.add_argument('--model', type=str, default='dare_0', help='model to test.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--criterion', type=str, default='gini', help='splitting criterion.')

    # tree/forest hyperparameters
    parser.add_argument('--max_features', type=str, default='sqrt', help='maximum no. features to sample.')

    # display settings
    parser.add_argument('--verbose', type=int, default=2, help='verbosity level.')

    args = parser.parse_args()
    main(args)
