"""
Cleaning experiment.
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


def flip_labels(arr, seed, k=100, logger=None):
    """
    Flips the label of random elements in an array; only for binary arrays.
    """
    assert arr.ndim == 1, 'arr is not 1d!'
    assert np.all(np.unique(arr) == np.array([0, 1])), 'arr is not binary!'

    if k <= 1.0:
        assert isinstance(k, float), 'k is not a float!'
        assert k > 0, 'k is less than zero!'

        k = int(len(arr) * k)

    assert k <= len(arr), 'k is greater than len(arr)!'

    np.random.seed(seed)
    noisy_indices = np.random.choice(np.arange(len(arr)), size=k, replace=False)

    new_arr = arr.copy()
    ones_flipped = 0
    zeros_flipped = 0

    for noisy_ndx in noisy_indices:
        if new_arr[noisy_ndx] == 1:
            ones_flipped += 1
        else:
            zeros_flipped += 1
        new_arr[noisy_ndx] = 0 if new_arr[noisy_ndx] == 1 else 1

    if logger:
        logger.info('sum before: {:,}'.format(np.sum(arr)))
        logger.info('ones flipped: {:,}'.format(ones_flipped))
        logger.info('zeros flipped: {:,}'.format(zeros_flipped))
        logger.info('sum after: {:,}'.format(np.sum(new_arr)))

    assert np.sum(new_arr) == np.sum(arr) - ones_flipped + zeros_flipped

    return new_arr, noisy_indices


def flip_labels_with_indices(arr, indices):
    """
    Flips the label of specified elements in an array; only for binary arrays.
    """
    assert arr.ndim == 1, 'arr is not 1d!'
    assert np.all(np.unique(arr) == np.array([0, 1])), 'arr is not binary!'

    new_arr = arr.copy()
    for ndx in indices:
        new_arr[ndx] = 0 if new_arr[ndx] == 1 else 1

    return new_arr


def measure_performance(args, checkpoints, fixed_indices, noisy_indices, model_noisy,
                        X_train, y_train, X_test, y_test, logger=None):
    """
    Retrains the tree ensemble for each ckeckpoint, where a checkpoint represents
    which flipped labels have been fixed.
    """

    begin = time.time()

    checked_pct = [0]
    fix_pct = [0]

    # totally noisy model
    auc, acc, ap = exp_util.performance(model_noisy, X_test, y_test,
                                        logger=logger, name='noisy')
    accs = [acc]
    aucs = [auc]
    aps = [ap]

    s = 'checkpoint {:>2}: no. checked: {:>7,}, no. fixed: {:>7,}, cum_time: {:7.1f}s'

    for i, (n_checked, n_fixed) in enumerate(checkpoints):
        label = s.format(i + 1, n_checked, n_fixed, time.time() - begin)
        fix_indices = fixed_indices[:n_fixed]

        # fix a portion of the noisy samples
        semi_noisy_indices = np.setdiff1d(noisy_indices, fix_indices)
        y_train_semi_noisy = flip_labels_with_indices(y_train, semi_noisy_indices)

        # train a new model on this partially fixed dataset
        model_semi_noisy = _get_model(args).fit(X_train, y_train_semi_noisy)
        auc, acc, ap = exp_util.performance(model_semi_noisy, X_test, y_test,
                                            logger=logger, name=label)

        accs.append(acc)
        aucs.append(auc)
        aps.append(ap)
        checked_pct.append(float(n_checked / len(y_train)))
        fix_pct.append(float(n_fixed / len(noisy_indices)))

    results = {}
    results['acc'] = accs
    results['auc'] = aucs
    results['ap'] = aps
    results['checked_pct'] = checked_pct

    return results


def record_fixes(train_order, noisy_indices, snapshot_interval):
    """
    Returns the number of train instances checked and which train instances were
    fixed for each checkpoint.
    """
    fixed_indices = []
    checkpoints = []
    checked = 0
    snapshot = 1

    # record which noisy training instances are fixed
    for train_ndx in train_order:
        if train_ndx in noisy_indices:
            fixed_indices.append(train_ndx)
        checked += 1

        # save snapshots of how many instances are checked and fixed
        if checked >= snapshot * snapshot_interval:
            checkpoints.append((checked, len(fixed_indices)))
            snapshot += 1

    return checkpoints, np.array(fixed_indices)


def experiment(args, logger, out_dir):
    """
    Obtains data, trains model, and generates instance-attribution explanations.
    """

    # get data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, data_dir=args.data_dir)

    logger.info('\nno. train instances: {:,}'.format(len(X_train)))
    logger.info('no. test instances: {:,}'.format(len(X_test)))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # add noise
    y_train_noisy, noisy_indices = flip_labels(y_train, seed=args.rs, k=args.flip_frac)
    noisy_indices = np.array(sorted(noisy_indices))
    logger.info('no. noisy labels: {:,}'.format(len(noisy_indices)))

    # number of checkpoints to record
    n_check = int(len(y_train) * args.check_frac)
    snapshot_interval = n_check / args.n_snapshots
    logger.info('no. check: {:,}'.format(n_check))
    logger.info('no. snapshots: {:,}'.format(args.n_snapshots))

    # experiment settings
    logger.info('\nrandom state: {}'.format(args.rs))
    logger.info('criterion: {}'.format(args.criterion))
    logger.info('n_estimators: {}'.format(args.n_estimators))
    logger.info('max_depth: {}'.format(args.max_depth))
    logger.info('max_features: {}\n'.format(args.max_features))

    # show model performance before and after noise
    model_clean = _get_model(args).fit(X_train, y_train)
    model_noisy = _get_model(args).fit(X_train, y_train_noisy)
    acc_clean, auc_clean, ap_clean = exp_util.performance(model_clean, X_test, y_test,
                                                          logger=logger, name='clean')
    acc_clean, auc_clean, ap_clean = exp_util.performance(model_noisy, X_test, y_test,
                                                          logger=logger, name='noisy')

    start = time.time()

    # random method
    if args.method == 'random':
        logger.info('\nOrdering by random...')

        # +1 to avoid choosing the same indices as the noisy labels
        np.random.seed(args.rs + 1)
        train_order = np.random.choice(len(y_train), size=n_check, replace=False)

    # D-DART
    elif args.method == 'dart':
        logger.info('\nOrdering by D-DART...')
        explanation = exp_util.explain(model_noisy, X_train, y_train, X_test)
        train_order = np.argsort(np.sum(np.abs(explanation), axis=1))[::-1]

    # D-DART
    elif args.method == 'dart_loss':
        logger.info('\nOrdering by D-DART loss...')
        proba = model_noisy.predict_proba(X_train)[:, 1]
        loss = np.abs(proba - y_train_noisy)
        train_order = np.argsort(loss)[::-1]

    # save results
    checkpoints, fixed_indices = record_fixes(train_order[:n_check], noisy_indices, snapshot_interval)
    results = measure_performance(args, checkpoints, fixed_indices, noisy_indices, model_noisy,
                                  X_train, y_train, X_test, y_test, logger=logger)
    np.save(os.path.join(out_dir, 'results.npy'), results)

    logger.info('time: {:3f}s'.format(time.time() - start))


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
    parser.add_argument('--method', type=str, default='dart', help='method to use.')
    parser.add_argument('--flip_frac', type=float, default=0.4, help='% of data to flip.')
    parser.add_argument('--check_frac', type=float, default=0.3, help='% of data to check.')
    parser.add_argument('--n_snapshots', type=int, default=10, help='no. points to record and plot.')

    # tree hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_depth', type=int, default=10, help='maximum depth of the tree.')
    parser.add_argument('--max_features', type=float, default=0.25, help='maximum features to sample.')
    parser.add_argument('--criterion', type=str, default='gini', help='splitting criterion.')

    args = parser.parse_args()
    main(args)
