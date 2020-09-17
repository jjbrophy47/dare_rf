"""
Model-aware adversary that choose each instance based on the cumulative
total number of samples used for retraining subtrees over the ensemble
when adding/deleting that sample.

This is a search problem, and is likely NP-hard; however, we can
approximate it by sampling a subset of instances, and picking the
worst one of those, and then repeating this process.
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
import dart_rf as dart
from baselines import cedar
from utility import data_util
from utility import exp_util
from utility import print_util


def _get_name(args):
    """
    Return name of method based on the settings.
    """
    result = ''
    if args.cedar:
        result = 'cedar'
    elif args.dart:
        result = 'dart'
        if args.topd == 0:
            result = 'exact'
    else:
        raise ValueError('no model selected!')
    return result


def _process_retrains(types, depths):
    """
    Compress the information about retrain depth into counts.
    """
    r = {k: defaultdict(int) for k in set(types)}

    for t, d in zip(types, depths):
        r[t][d] += 1

    return r


def _get_model(args):
    """
    Return model.
    """
    if args.cedar:
        model = cedar.Forest(epsilon=args.epsilon,
                             lmbda=args.lmbda,
                             max_depth=args.max_depth,
                             criterion=args.criterion,
                             topd=args.topd,
                             min_support=args.min_support,
                             n_estimators=args.n_estimators,
                             max_features=args.max_features,
                             verbose=args.verbose,
                             random_state=args.rs)

    elif args.dart:
        model = dart.Forest(max_depth=args.max_depth,
                            criterion=args.criterion,
                            topd=args.topd,
                            min_support=args.min_support,
                            n_estimators=args.n_estimators,
                            max_features=args.max_features,
                            verbose=args.verbose,
                            random_state=args.rs)
    else:
        raise ValueError('no model selected!')

    return model


def _get_naive(args):
    """
    Return naive model.
    """
    model = dart.Forest(max_depth=args.max_depth,
                        criterion=args.criterion,
                        topd=0,
                        min_support=args.min_support,
                        n_estimators=args.n_estimators,
                        max_features=args.max_features,
                        verbose=args.verbose,
                        random_state=args.rs)
    return model


def get_allotted_time(args, X_train, y_train, X_test, y_test, logger=None):
    """
    Compute the time it takes to add/delete a specified number of
    smaples frmo a naive model sequentially.
    """

    # initial naive train
    model = _get_naive(args)
    start = time.time()
    model = model.fit(X_train, y_train)
    train_time = time.time() - start

    logger.info('\n[{}] train time: {:.3f}s'.format('naive1', train_time))
    auc, acc, ap = exp_util.performance(model, X_test, y_test,
                                        logger=None, name='naive')

    # naive train after adding/deleting data
    np.random.seed(args.rs)
    delete_indices = np.random.choice(np.arange(X_train.shape[0]),
                                      size=args.n_update, replace=False)

    if args.operation == 'delete':
        new_X_train = np.delete(X_train, delete_indices, axis=0)
        new_y_train = np.delete(y_train, delete_indices)
    else:
        new_X_train = np.vstack([X_train, X_train[delete_indices]])
        new_y_train = np.concatenate([y_train, y_train[delete_indices]])

    model = _get_naive(args)
    start = time.time()
    model = model.fit(new_X_train, new_y_train)
    update_time = time.time() - start

    logger.info('[{}] train time: {:.3f}s'.format('naive2', train_time))

    # interpolate sequential updates
    total_time = ((train_time + update_time) / 2) * args.n_update
    initial_utility = auc, acc, ap

    return total_time, initial_utility


def get_update_ndx(model, X_train, y_train, indices):
    """
    Select a sample based on the subset of samples.
    """
    start = time.time()

    np.random.seed(args.rs)
    subsample_indices = np.random.choice(indices, size=args.subsample_size, replace=False)

    # test subsamples
    ndx_cost_list = []
    for j, subsample_ndx in enumerate(subsample_indices):

        if args.operation == 'delete':

            # delete test sample
            model.delete(subsample_ndx, sim_mode=True)

            sample_cost = model.get_removal_retrain_sample_count()
            ndx_cost_list.append((subsample_ndx, sample_cost))
            model.clear_removal_metrics()

            # add sample back in to reset for the next test sample
            model.add(X_train[[subsample_ndx]], y_train[[subsample_ndx]], sim_mode=True)
            model.clear_add_metrics()

        # add operation
        else:
            # delete test sample
            add_indices = model.add(X_train[[subsample_ndx]], y_train[[subsample_ndx]],
                                    get_indices=True, sim_mode=True)

            sample_cost = model.get_add_retrain_sample_count()
            ndx_cost_list.append((subsample_ndx, sample_cost))
            model.clear_add_metrics()

            # delete sample to reset for the next test sample
            model.delete(add_indices, sim_mode=True)
            model.clear_removal_metrics()

    # extract best sample based on update time
    ndx_cost_list = sorted(ndx_cost_list, key=lambda x: x[1])[::-1]
    best_ndx = ndx_cost_list[0][0]

    search_time = time.time() - start

    return best_ndx, search_time


def update_model(model, X_train, y_train, update_ndx, allotted_time):
    """
    Add or delete the specified sample.
    """

    if args.operation == 'delete':
        start = time.time()
        model.delete(update_ndx)
        update_time = time.time() - start
        sample_cost = model.get_removal_retrain_sample_count()
        retrain_depths = model.get_removal_retrain_depths()
        model.clear_removal_metrics()

    else:
        start = time.time()
        model.add(X_train[[update_ndx]], y_train[[update_ndx]])
        update_time = time.time() - start
        sample_cost = model.get_add_retrain_sample_count()
        retrain_depths = model.get_add_retrain_depths()
        model.clear_add_metrics()

    return model, update_time, sample_cost, retrain_depths


def check_utility(model, X_train, y_train, X_test, y_test,
                  updated_indices, logger=None, name=''):
    """
    Utility check on the model and exact model.
    """
    logger.info('testing {} and exact utility...'.format(name))

    # check model performance
    auc_model, acc_model, ap_model = exp_util.performance(model, X_test, y_test,
                                                          logger=None, name=name)
    result = (auc_model, acc_model, ap_model)

    start = time.time()

    if args.operation == 'delete':
        new_X_train = np.delete(X_train, updated_indices, axis=0)
        new_y_train = np.delete(y_train, updated_indices)
    else:
        new_X_train = np.vstack([X_train, X_train[updated_indices]])
        new_y_train = np.concatenate([y_train, y_train[updated_indices]])

    exact_model = _get_naive(args)
    exact_model = exact_model.fit(new_X_train, new_y_train)
    end = time.time() - start

    logger.info('[exact] updating batch of {}: {:.3f}s'.format(len(updated_indices), end))
    auc_exact, acc_exact, ap_exact = exp_util.performance(exact_model, X_test, y_test,
                                                          logger=None, name='exact')
    result += (auc_exact, acc_exact, ap_exact)

    return result


def select_samples(args, X_train, y_train, X_test, y_test,
                   allotted_time, initial_utility, logger=None):
    """
    Search for n_samples samples based on the cumulative
    number of instances involved in retraining in the ensemble.
    """
    begin = time.time()

    # result containers
    starting_allotted_time = allotted_time
    total_update_time = 0
    retrain_types_list = []
    retrain_depths_list = []

    # train target model
    model = _get_model(args)
    name = _get_name(args)

    start = time.time()
    model = model.fit(X_train, y_train)
    train_time = time.time() - start

    logger.info('[{}] train time: {:.3f}s'.format(name, train_time))

    auc_exact, acc_exact, ap_exact = initial_utility
    auc_model, acc_model, ap_model = exp_util.performance(model, X_test, y_test,
                                                          logger=logger, name=name)

    # result containers
    aucs_model, accs_model, aps_model = [auc_model], [acc_model], [ap_model]
    aucs_exact, accs_exact, aps_exact = [auc_exact], [acc_exact], [ap_exact]

    # index trackers
    indices = np.arange(len(X_train))
    updated_indices = []

    # find the most damaging samples heuristically
    i = 0
    j = 0
    while allotted_time > 0 and time.time() - begin <= args.time_limit:

        update_ndx, search_time = get_update_ndx(model, X_train, y_train, indices)
        model, update_time, sample_cost, retrain_stats = update_model(model, X_train, y_train,
                                                                      update_ndx, allotted_time)

        total_update_time += update_time
        allotted_time -= update_time
        cum_time = time.time() - begin
        retrain_types_list.append(retrain_stats[0])
        retrain_depths_list.append(retrain_stats[1])

        s = '[{}] sample {}   sample_cost: {:,}   search: {:3f}s   allotted: {:.3f}s'
        s += '   cum time: {:.3f}s'
        logger.info(s.format(i + 1, update_ndx, sample_cost, search_time, allotted_time, cum_time))

        # remove the chosen ndx from the list of available indices
        indices = np.setdiff1d(indices, [update_ndx])
        updated_indices.append(update_ndx)

        if i == args.n_check:
            check_point = int((starting_allotted_time / (total_update_time / len(updated_indices))) / args.n_check)
            logger.info('checkpoint established, no. points: {}'.format(check_point))
            j = 0

        # record performance
        if i >= args.n_check and j == check_point:
            result = check_utility(model, X_train, y_train, X_test, y_test,
                                   updated_indices, logger=logger, name=name)
            auc_model, acc_model, ap_model, auc_exact, acc_exact, ap_exact = result

            l1 = [auc_model, acc_model, ap_model, auc_exact, acc_exact, ap_exact]
            l2 = [aucs_model, accs_model, aps_model, aucs_exact, accs_exact, aps_exact]
            [l.append(x) for x, l in zip(l1, l2)]

            check_point = int((starting_allotted_time / (total_update_time / len(updated_indices))) / args.n_check)
            logger.info('checkpoint re-established, no. points: {}'.format(check_point))
            j = 0

        i += 1
        j += 1

    # estimate how many additional updates would finish in the remaining time
    if allotted_time > 0:
        average_updated_time = total_update_time / i
        i += int(allotted_time) / average_updated_time

    # get model statistics
    n_nodes_avg, n_exact_avg, n_semi_avg = model.get_node_statistics()
    retrain_types = np.concatenate(retrain_types_list)
    retrain_depths = np.concatenate(retrain_depths_list)

    # save model results
    results = model.get_params()
    results['allotted_time'] = starting_allotted_time
    results['n_naive'] = args.n_update
    results['n_model'] = i
    results['percent_complete'] = i / len(X_train)
    results['retrains'] = _process_retrains(retrain_types, retrain_depths)
    results['auc'] = np.array(aucs_model)
    results['acc'] = np.array(accs_model)
    results['ap'] = np.array(aps_model)
    results['exact_auc'] = np.array(aucs_exact)
    results['exact_acc'] = np.array(accs_exact)
    results['exact_ap'] = np.array(aps_exact)
    results['n_nodes_avg'] = n_nodes_avg
    results['n_exact_avg'] = n_exact_avg
    results['n_semi_avg'] = n_semi_avg
    results['model'] = name

    return results


def experiment(args, logger, out_dir, seed):
    """
    Obtains data, trains model, and computes adversaral instances.
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
    logger.info('min_support: {}'.format(args.min_support))
    logger.info('epsilon: {}'.format(args.epsilon))
    logger.info('lmbda: {}'.format(args.lmbda))
    logger.info('subsample_size: {}'.format(args.subsample_size))
    logger.info('n_update: {}'.format(args.n_update))
    logger.info('n_check: {}'.format(args.n_check))

    allotted_time, initial_utility = get_allotted_time(args, X_train, y_train, X_test, y_test,
                                                       logger=logger)

    results = select_samples(args, X_train, y_train, X_test, y_test,
                             allotted_time=allotted_time, initial_utility=initial_utility,
                             logger=logger)

    logger.info('{}'.format(results))
    np.save(os.path.join(out_dir, 'results.npy'), results)


def main(args):

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.operation,
                           args.criterion,
                           'rs_{}'.format(args.rs),
                           'trees_{}'.format(args.n_estimators),
                           'depth_{}'.format(args.max_depth),
                           'features_{}'.format(args.max_features),
                           'sub_{}'.format(args.subsample_size))

    if args.dart:
        out_dir = os.path.join(out_dir,
                               'topd_{}'.format(args.topd),
                               'support_{}'.format(args.min_support))

    elif args.cedar:
        out_dir = os.path.join(out_dir,
                               'topd_{}'.format(args.topd),
                               'support_{}'.format(args.min_support),
                               'epsilon_{}'.format(args.epsilon),
                               'lmbda_{}'.format(args.lmbda))

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
    parser.add_argument('--out_dir', type=str, default='output/update/', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='surgical', help='dataset to use for the experiment.')
    parser.add_argument('--append_results', action='store_true', default=False, help='add results.')

    # experiment settings
    parser.add_argument('--rs', type=int, default=1, help='seed to enhance reproducibility.')
    parser.add_argument('--operation', type=str, default='delete', help='add or delete.')
    parser.add_argument('--n_check', type=int, default=5, help='check utility after no. instances.')
    parser.add_argument('--n_update', type=int, default=1, help='number of instances for naive to add/delete.')
    parser.add_argument('--time_limit', type=int, default=72000, help='seconds given for the entire experiment.')

    # tree hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_depth', type=int, default=10, help='maximum depth of the tree.')
    parser.add_argument('--max_features', type=float, default=0.25, help='maximum features to sample.')
    parser.add_argument('--criterion', type=str, default='gini', help='splitting criterion.')

    # DART
    parser.add_argument('--dart', action='store_true', default=False, help='DART model.')
    parser.add_argument('--topd', type=int, default=0, help='no. top layers to be random.')
    parser.add_argument('--min_support', type=int, default=2, help='minimum number of samples for stochastic.')

    # CEDAR
    parser.add_argument('--cedar', action='store_true', default=False, help='CEDAR model.')
    parser.add_argument('--epsilon', type=float, default=1.0, help='setting for certified adversarial ordering.')
    parser.add_argument('--lmbda', type=float, default=-1, help='noise hyperparameter.')

    # adversary settings
    parser.add_argument('--subsample_size', type=int, default=1, help='number samples to test at a time.')

    # display settings
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level.')

    args = parser.parse_args()
    main(args)
