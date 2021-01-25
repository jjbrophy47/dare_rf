"""
Model-aware adversary that choose each instance based on the cumulative
total number of samples used for retraining subtrees over the ensemble
when deleting that sample.

This is a search problem, and is likely NP-hard; however, we can
approximate it by sampling a subset of instances, and picking the
worst one of those, then repeating this process.
"""
import os
import sys
import time
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import dart
from utility import data_util
from utility import exp_util
from utility import print_util


def count_depths(types, depths):
    """
    Compress the information about deletion types and depths into counts.
    """

    # get list of deletion types
    r = {k: defaultdict(int) for k in set(types)}

    # count no. deletions at each depth for each deletion type
    for t, d in zip(types, depths):
        r[t][d] += 1

    # convert defaultdicts to regular dicts
    for k in r.keys():
        r[k] = dict(r[k])

    return r

def count_costs(types, depths, costs):
    """
    For retrains (types = 1), compute the total cost
    for each depth
    """

    # only use indices where a retrain occurred
    retrain_indices = np.where(types == 1)[0]

    # get list of all retrain depths
    r = {d: 0 for d in set(depths[retrain_indices])}

    # compute total cost for each depth
    for d, c in zip(depths[retrain_indices], costs[retrain_indices]):
        r[d] += c

    return r


def get_model(args):
    """
    Return model.
    """
    model = dart.Forest(max_depth=args.max_depth,
                        criterion=args.criterion,
                        topd=args.topd,
                        k=args.k,
                        n_estimators=args.n_estimators,
                        max_features=args.max_features,
                        verbose=args.verbose,
                        random_state=args.rs)

    return model


def get_naive(args):
    """
    Return naive model.
    """
    model = dart.Forest(max_depth=args.max_depth,
                        criterion=args.criterion,
                        topd=0,
                        k=args.k,
                        n_estimators=args.n_estimators,
                        max_features=args.max_features,
                        verbose=args.verbose,
                        random_state=args.rs)
    return model


def train_naive(args, X_train, y_train, X_test, y_test, logger=None):
    """
    Compute the time it takes to delete a specified number of
    samples from a naive model sequentially.
    """

    # initial naive training time
    model = get_naive(args)
    start = time.time()
    model = model.fit(X_train, y_train)
    before_train_time = time.time() - start
    logger.info('\n[{}] before train time: {:.3f}s'.format('naive', before_train_time))

    # predictive performance of the naive model
    auc, acc, ap = exp_util.performance(model, X_test, y_test, logger=logger, name='naive')

    # naive train after deleting data
    np.random.seed(args.rs)
    delete_indices = np.random.choice(np.arange(X_train.shape[0]),
                                      size=args.n_delete, replace=False)
    new_X_train = np.delete(X_train, delete_indices, axis=0)
    new_y_train = np.delete(y_train, delete_indices)

    # after training time
    model = get_naive(args)
    start = time.time()
    model = model.fit(new_X_train, new_y_train)
    after_train_time = time.time() - start
    logger.info('[{}] after train time: {:.3f}s'.format('naive', after_train_time))

    # interpolate sequential updates
    total_time = ((before_train_time + after_train_time) / 2) * args.n_delete
    initial_utility = auc, acc, ap

    return total_time, initial_utility


def get_delete_index(model, X_train, y_train, indices):
    """
    Randomly select a subset of samples, simulate deleting each one,
    then pick the sample that causes the largest no. samples to be retrained.
    """
    start = time.time()

    # randomly samples a subset of indices to simulate deleting
    np.random.seed(args.rs)
    subsample_indices = np.random.choice(indices, size=args.subsample_size, replace=False)

    # return the only sample if the subset size is 1
    if args.subsample_size == 1:
        return subsample_indices[0], time.time() - start

    # simulate deleting samples
    best_ndx = -1
    best_score = -1
    for j, subsample_ndx in enumerate(subsample_indices):

        # simulate deletion
        sample_cost = model.sim_delete(subsample_ndx)

        # save best sample
        if sample_cost > best_score:
            best_ndx = subsample_ndx
            best_score = sample_cost

    # record search time
    search_time = time.time() - start

    return best_ndx, search_time


def experiment(args, logger, out_dir, seed):
    """
    Delete as many samples in the time it takes the naive
    approach to delete one sample.
    """

    # get data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, data_dir=args.data_dir)

    # dataset statistics
    logger.info('\ntrain instances: {:,}'.format(X_train.shape[0]))
    logger.info('test instances: {:,}'.format(X_test.shape[0]))
    logger.info('features: {:,}'.format(X_train.shape[1]))

    # experiment settings
    logger.info('\nrandom state: {}'.format(seed))
    logger.info('criterion: {}'.format(args.criterion))
    logger.info('n_estimators: {}'.format(args.n_estimators))
    logger.info('max_depth: {}'.format(args.max_depth))
    logger.info('topd: {}'.format(args.topd))
    logger.info('k: {}'.format(args.k))
    logger.info('subsample_size: {}'.format(args.subsample_size))
    logger.info('n_delete: {}'.format(args.n_delete))

    # train a naive model, before and after deleting 1 sample
    naive_avg_delete_time, naive_utility = train_naive(args, X_train, y_train, X_test, y_test, logger=logger)

    # begin experiment
    begin = time.time()

    # amount of time given to delete as many samples as possible
    allotted_time = naive_avg_delete_time

    # result containers
    total_delete_time = 0
    delete_types_list = []
    delete_depths_list = []
    delete_costs_list = []

    # train target model
    model = get_model(args)

    start = time.time()
    model = model.fit(X_train, y_train)
    train_time = time.time() - start

    logger.info('[{}] train time: {:.3f}s'.format('model', train_time))

    # evaluate predictive performance between naive and the model
    naive_auc, naive_acc, naive_ap = naive_utility
    model_auc, model_acc, model_ap = exp_util.performance(model, X_test, y_test, logger=logger, name='model')

    # available indices
    indices = np.arange(len(X_train))

    # find the most damaging samples heuristically
    progress_str = '[{}] sample {}, sample_cost: {:,}, search time: {:3f}s, allotted: {:.3f}s, cum time: {:.3f}s'
    logger.info('\nDelete samples:')

    n_deleted = 0
    while allotted_time > 0 and time.time() - begin <= args.time_limit:

        # adversarially select a sample out of a subset of candidate samples
        delete_ndx, search_time = get_delete_index(model, X_train, y_train, indices)

        # delete the adversarially selected sample
        start = time.time()
        model.delete(delete_ndx)
        delete_time = time.time() - start

        # get deletion statistics
        delete_types, delete_depths, delete_costs = model.get_delete_metrics()
        delete_types_list.append(delete_types)
        delete_depths_list.append(delete_depths)
        delete_costs_list.append(delete_costs)
        sample_cost = np.sum(delete_costs)  # sum over all trees
        model.clear_delete_metrics()

        # update counters
        allotted_time -= delete_time  # available time
        total_delete_time += delete_time  # total deletion time
        cum_time = time.time() - begin  # total time
        n_deleted += 1

        # progress update
        logger.info(progress_str.format(n_deleted, delete_ndx, sample_cost, search_time, allotted_time, cum_time))

        # remove the chosen ndx from the list of available indices
        indices = np.setdiff1d(indices, [delete_ndx])

    # estimate how many additional updates would finish in the remaining time
    if allotted_time > 0:
        average_delete_time = total_delete_time / n_deleted
        n_deleted += int(allotted_time) / average_delete_time

    # get model statistics
    # n_nodes_avg, n_exact_avg, n_semi_avg = model.get_node_statistics()
    delete_types = np.concatenate(delete_types_list)
    delete_depths = np.concatenate(delete_depths_list)
    delete_costs = np.concatenate(delete_costs_list)

    # save model results
    results = model.get_params()
    results['naive_auc'] = naive_auc
    results['naive_acc'] = naive_acc
    results['naive_ap'] = naive_ap
    results['naive_avg_delete_time'] = naive_avg_delete_time
    results['naive_n_deleted'] = args.n_delete
    results['model_n_deleted'] = n_deleted
    results['model_train_%_deleted'] = n_deleted / len(X_train)
    results['model_delete_depths'] = count_depths(delete_types, delete_depths)
    results['model_delete_costs'] = count_costs(delete_types, delete_depths, delete_costs)
    results['model_auc'] = model_auc
    results['model_acc'] = model_acc
    results['model_ap'] = model_ap
    # results['model_n_avg_nodes'] = n_nodes_avg
    # results['model_n_exact_avg'] = n_exact_avg
    # results['model_n_semi_avg'] = n_semi_avg

    logger.info('\nResults:\n{}'.format(results))
    np.save(os.path.join(out_dir, 'results.npy'), results)

    return results


def main(args):

    # assertions
    assert args.criterion in ['gini', 'entropy']

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.criterion,
                           'rs_{}'.format(args.rs),
                           'trees_{}'.format(args.n_estimators),
                           'depth_{}'.format(args.max_depth),
                           'topd_{}'.format(args.topd),
                           'k_{}'.format(args.k),
                           'sub_{}'.format(args.subsample_size))

    log_fp = os.path.join(out_dir, 'log.txt')
    os.makedirs(out_dir, exist_ok=True)

    # skip experiment if results already exist
    if args.append_results and os.path.exists(os.path.join(out_dir, 'results.npy')):
        print('results exist: {}'.format(out_dir))
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
    parser.add_argument('--out_dir', type=str, default='output/delete/', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='surgical', help='dataset to use for the experiment.')
    parser.add_argument('--append_results', action='store_true', default=False, help='add results.')

    # experiment settings
    parser.add_argument('--rs', type=int, default=1, help='seed to enhance reproducibility.')
    parser.add_argument('--n_delete', type=int, default=1, help='number of instances for naive to delete.')
    parser.add_argument('--time_limit', type=int, default=72000, help='seconds given for the entire experiment.')

    # tree hyperparameters
    parser.add_argument('--criterion', type=str, default='gini', help='gini or entropy.')
    parser.add_argument('--n_estimators', type=int, default=100, help='no. trees in the forest.')
    parser.add_argument('--max_depth', type=int, default=10, help='max. depth of the tree.')
    parser.add_argument('--max_features', type=str, default='sqrt', help='maximum no. features to sample.')

    # DART
    parser.add_argument('--topd', type=int, default=0, help='no. top layers to be random.')
    parser.add_argument('--k', type=int, default=10, help='no. thresholds to sample.')

    # adversary settings
    parser.add_argument('--subsample_size', type=int, default=1, help='number samples to test at a time.')

    # display settings
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level.')

    args = parser.parse_args()
    main(args)
