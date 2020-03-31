"""
Experiment: Analyzes the amortized runtime under different deletion orerings.
"""
import os
import sys
import time
import argparse
from collections import Counter

import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../..')
sys.path.insert(0, here + '/../..')
import cedar
from utility import data_util, exp_util, print_util, exact_adv_util, cert_adv_util


def _adjust_indices(indices_to_delete):
    """
    Return an adjusted array of indices, taking into account removing each one sequentially.

    Example:
    indices_to_delete = [4, 1, 10, 0] => desired result = [4, 1, 8, 0]
    """
    assert len(np.unique(indices_to_delete)) == len(indices_to_delete)
    indices = indices_to_delete.copy()
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if indices[i] < indices[j]:
                indices[j] -= 1
    return indices


def unlearning_method(args, seed, out_dir, logger, X_train, y_train, X_test, y_test,
                      delete_indices, name):

    epsilon = 0 if name == 'exact' else args.epsilon
    lmbda = -1 if name == 'exact' else args.lmbda

    # train
    logger.info('\n{}'.format(name.capitalize()))
    start1 = time.time()
    model = cedar.Forest(epsilon=epsilon, lmbda=lmbda,
                         n_estimators=args.n_estimators, max_features=args.max_features,
                         max_depth=args.max_depth, verbose=args.verbose, random_state=seed)
    model = model.fit(X_train, y_train)
    end_time = time.time() - start1
    logger.info('[{}] train time: {:.3f}s'.format(name, end_time))
    auc, acc = exp_util.performance(model, X_test, y_test, logger=logger, name=name)
    performance_markers = [int(x) for x in np.linspace(0, len(delete_indices) - 1, 10)]

    # delete
    times = [end_time]
    aucs, accs = [auc], [acc]
    for i, delete_ndx in enumerate(delete_indices):

        start = time.time()
        model.delete(delete_ndx)
        end_time = time.time() - start
        times.append(end_time)

        if i in performance_markers:
            auc, acc = exp_util.performance(model, X_test, y_test, logger=None, name=name)
            aucs.append(auc)
            accs.append(acc)

            if args.verbose > 0:
                logger.info('{}: cumulative run time: {:.3f}s'.format(i, time.time() - start1))

    types, depths = model.get_removal_statistics()
    types_counter = Counter(types)
    depths_counter = Counter(depths)
    logger.info('[{}] amortized: {:.3f}s'.format(name, np.mean(times)))
    logger.info('[{}] types: {}'.format(name, types_counter))
    logger.info('[{}] depths: {}'.format(name, depths_counter))
    auc, acc = exp_util.performance(model, X_test, y_test, logger=logger, name=name)

    if args.save_results:
        d = model.get_params()
        d['time'] = np.array(times)
        d['type'] = np.array(types)
        d['depth'] = np.array(depths)
        d['auc'] = np.array(aucs)
        d['acc'] = np.array(accs)
        np.save(os.path.join(out_dir, '{}.npy'.format(name)), d)


def naive_method(args, seed, out_dir, logger, X_train, y_train, X_test, y_test, delete_indices, name):

    adjusted_indices = _adjust_indices(delete_indices)
    n_remove = len(delete_indices)

    # train
    logger.info('\n{}'.format(name.capitalize()))
    start = time.time()
    model = cedar.Forest(epsilon=0, lmbda=-1,
                         n_estimators=args.n_estimators, max_features=args.max_features,
                         max_depth=args.max_depth, verbose=args.verbose, random_state=seed)
    model = model.fit(X_train, y_train)
    end_time = time.time() - start
    logger.info('[{}] train time: {:.3f}s'.format(name, end_time))
    auc, acc = exp_util.performance(model, X_test, y_test, logger=logger, name=name)

    # actually retrain for each deletion
    if args.retrain:

        # deleting (i.e. retraining)
        times = [end_time]
        X_train_new, y_train_new = X_train.copy(), y_train.copy()
        for i, delete_ndx in enumerate(adjusted_indices):

            X_train_new = np.delete(X_train_new, delete_ndx, axis=0)
            y_train_new = np.delete(y_train_new, delete_ndx)

            start = time.time()
            model = cedar.Forest(epsilon=0, lmbda=-1,
                                 n_estimators=args.n_estimators, max_features=args.max_features,
                                 max_depth=args.max_depth, verbose=args.verbose, random_state=seed)
            model = model.fit(X_train, y_train)
            end_time = time.time() - start
            times.append(end_time)

            if args.verbose > 0 and i % 100 == 0:
                logger.info('{}. [{}] retrain time: {:.3f}s'.format(i, delete_ndx, end_time))

    # train again after deleting all instances and draw a line between the two points
    else:
        X_train_new = np.delete(X_train, delete_indices, axis=0)
        y_train_new = np.delete(y_train, delete_indices)
        initial_train_time = end_time
        start = time.time()
        model = cedar.Forest(epsilon=0, lmbda=-1,
                             n_estimators=args.n_estimators, max_features=args.max_features,
                             max_depth=args.max_depth, verbose=args.verbose, random_state=seed)
        model = model.fit(X_train_new, y_train_new)
        last_train_time = time.time() - start
        times = np.linspace(initial_train_time, last_train_time, n_remove - 1)

    logger.info('[{}] amortized: {:.3f}s'.format(name, np.mean(times)))
    exp_util.performance(model, X_test, y_test, logger=logger, name=name)

    if args.save_results:
        d = model.get_params()
        d['time'] = np.array(times)
        d['n_train'] = X_train.shape[0]
        d['n_features'] = X_train.shape[1]
        np.save(os.path.join(out_dir, 'naive.npy'), d)


def remove_sample(args, logger, out_dir, seed):

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, seed, data_dir=args.data_dir)

    # dataset statistics
    logger.info('train instances: {}'.format(X_train.shape[0]))
    logger.info('test instances: {}'.format(X_test.shape[0]))
    logger.info('attributes: {}'.format(X_train.shape[1]))

    # choose instances to delete
    n_remove = args.n_remove if args.frac_remove is None else int(X_train.shape[0] * args.frac_remove)
    if args.adversary == 'random':
        np.random.seed(seed)
        delete_indices = np.random.choice(X_train.shape[0], size=n_remove, replace=False)
    if args.adversary == 'root':
        delete_indices = exact_adv_util.exact_adversary(X_train, y_train, n_samples=n_remove, seed=seed,
                                                        verbose=args.verbose, logger=logger)
    elif args.adversary == 'certified':
        delete_indices = cert_adv_util.certified_adversary(X_train, y_train, epsilon=args.epsilon, lmbda=args.lmbda,
                                                           n_samples=n_remove, seed=seed,
                                                           verbose=args.verbose, logger=logger)
    else:
        exit('unknown adversary: {}'.format(args.adversary))

    logger.info('instances to delete: {}'.format(len(delete_indices)))
    logger.info('adversary: {}'.format(args.adversary))

    # naive retraining method
    if args.naive:
        naive_method(args, seed, out_dir, logger, X_train, y_train,
                     X_test, y_test, delete_indices, 'naive')

    # exact unlearning method
    if args.exact:
        unlearning_method(args, seed, out_dir, logger, X_train, y_train,
                          X_test, y_test, delete_indices, 'exact')

    # approximate unlearning method
    if args.cedar:
        unlearning_method(args, seed, out_dir, logger, X_train, y_train,
                          X_test, y_test, delete_indices, 'cedar')


def main(args):

    # run experiment multiple times
    for i in range(args.repeats):

        # create output dir
        rs_dir = os.path.join(args.out_dir, args.dataset, '{}'.format(args.adversary), 'rs{}'.format(args.rs))
        os.makedirs(rs_dir, exist_ok=True)

        # create logger
        logger = print_util.get_logger(os.path.join(rs_dir, 'log.txt'.format(args.dataset)))
        logger.info(args)
        logger.info('\nRun {}, seed: {}'.format(i + 1, args.rs))

        # run experiment
        remove_sample(args, logger, rs_dir, seed=args.rs)
        args.rs += 1

        # remove logger
        print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='output/forest/amortize', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='synthetic', help='dataset to use for the experiment.')
    parser.add_argument('--rs', type=int, default=1, help='seed to enhance reproducibility.')
    parser.add_argument('--repeats', type=int, default=1, help='number of times to repeat the experiment.')

    parser.add_argument('--naive', action='store_true', default=False, help='Include retrain baseline.')
    parser.add_argument('--retrain', action='store_true', default=False, help='Retrain every time.')

    parser.add_argument('--exact', action='store_true', default=False, help='Include deterministic baseline.')

    parser.add_argument('--cedar', action='store_true', default=False, help='Include cedar model.')
    parser.add_argument('--epsilon', type=float, default=0.1, help='setting for certified adversarial ordering.')
    parser.add_argument('--lmbda', type=float, default=100, help='amount of noise to add to the model.')

    parser.add_argument('--n_remove', type=int, default=10, help='number of instances to sequentially delete.')
    parser.add_argument('--frac_remove', type=float, default=0.1, help='fraction of instances to delete.')
    parser.add_argument('--adversary', type=str, default='random', help='type of adversarial ordering.')

    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_features', type=float, default=None, help='maximum features to sample.')
    parser.add_argument('--max_depth', type=int, default=50, help='maximum depth of the tree.')

    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    parser.add_argument('--save_results', action='store_true', default=False, help='save results.')
    args = parser.parse_args()
    main(args)
