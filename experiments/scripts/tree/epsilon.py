"""
Experiment: Show how test accuracy changes when there are
            no retrainings (epsilon = lambda).
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


def vary_epsilon(args, logger, out_dir, seed):

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, seed, data_dir=args.data_dir)

    # dataset statistics
    logger.info('train instances: {:,}'.format(X_train.shape[0]))
    logger.info('test instances: {:,}'.format(X_test.shape[0]))
    logger.info('attributes: {:,}'.format(X_train.shape[1]))

    logger.info('\nExact')
    start = time.time()
    model = cedar.Tree(lmbda=-1, max_depth=args.max_depth, verbose=args.verbose, random_state=seed)
    model = model.fit(X_train, y_train)
    logger.info('{:.3f}s'.format(time.time() - start))

    exp_util.performance(model, X_test, y_test, name='exact')

    # save deterministic results
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

    # observe change in test acuracy as lambda varies
    logger.info('\nCeDAR')
    epsilons = np.linspace(args.min_epsilon, args.max_epsilon, args.n_epsilon)

    n_remove = args.n_remove if args.frac_remove is None else int(X_train.shape[0] * args.frac_remove)
    gamma = n_remove / X_train.shape[0]
    logger.info('n_remove: {}, gamma: {:e}'.format(n_remove, gamma))

    aucs, accs = [], []
    lmbdas = []
    random_state = exp_util.get_random_state(seed)
    for i, epsilon in enumerate(epsilons):
        lmbda = epsilon / gamma

        start = time.time()
        model = cedar.Tree(epsilon=epsilon, lmbda=lmbda, max_depth=args.max_depth,
                           verbose=args.verbose, random_state=random_state)
        model = model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, pred)

        out_str = '[{}] epsilon: {:e}, lmbda: {:e} => acc: {:.3f}, auc: {:.3f}'
        logger.info(out_str.format(i, epsilon, lmbda, acc, auc))

        aucs.append(auc)
        accs.append(acc)
        lmbdas.append(lmbda)

        if args.verbose > 0:
            exp_util.performance(model, X_test, y_test, name='cedar', logger=logger)

    if args.save_results:
        d = model.get_params()
        d['auc'] = aucs
        d['acc'] = accs
        d['epsilon'] = epsilons
        d['lmbda'] = lmbdas
        d['n_train'] = X_train.shape[0]
        d['n_features'] = X_train.shape[1]
        d['n_remove'] = n_remove
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
        vary_epsilon(args, logger, ep_dir, seed=args.rs)
        args.rs += 1

        # remove logger
        print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='output/tree/epsilon', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='synthetic', help='dataset to use for the experiment.')

    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--repeats', type=int, default=1, help='number of times to perform the experiment.')
    parser.add_argument('--save_results', action='store_true', default=False, help='save results.')

    parser.add_argument('--min_epsilon', type=float, default=0, help='minimum lambda.')
    parser.add_argument('--max_epsilon', type=float, default=2, help='maximum lambda.')
    parser.add_argument('--n_epsilon', type=int, default=10, help='number of data points.')

    parser.add_argument('--n_remove', type=int, default=1, help='number of instances to sequentially delete.')
    parser.add_argument('--frac_remove', type=float, default=None, help='fraction of instances to delete.')

    parser.add_argument('--max_depth', type=int, default=1, help='maximum depth of the tree.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    main(args)
