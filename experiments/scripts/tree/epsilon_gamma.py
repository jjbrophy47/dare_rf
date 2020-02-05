"""
This experiment tests the accuracy of the decision trees.
BABC: Binary Attributes Binary Classification.
"""
import os
import sys
import time
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
from mulan.trees.babc_tree_d import BABC_Tree_D
from mulan.trees.tree import Tree
from utility import data_util, exp_util, print_util


def vary_gamma(args, logger, out_dir, seed):

    # obtain data
    data = data_util.get_data(args.dataset, args.rs, data_dir=args.data_dir, n_samples=args.n_samples,
                              n_attributes=args.n_attributes, test_frac=args.test_frac, convert=False)
    X_train, X_test, y_train, y_test = data

    # dataset statistics
    logger.info('train instances: {}'.format(X_train.shape[0]))
    logger.info('test instances: {}'.format(X_test.shape[0]))
    logger.info('attributes: {}'.format(X_train.shape[1]))

    logger.info('building d tree...')
    start = time.time()
    td = BABC_Tree_D(max_depth=args.max_depth, verbose=args.verbose).fit(X_train, y_train)
    logger.info('{:.3f}s'.format(time.time() - start))

    exp_util.performance(td, X_test, y_test, name='tree_D')

    aucs, accs = [], []
    gammas = np.linspace(0.01, 0.99)
    for gamma in gammas:
        logger.info('\nbuilding DeTRACE with epsilon: {}, gamma: {}...'.format(args.epsilon, gamma))
        start = time.time()
        detrace = Tree(epsilon=args.epsilon, gamma=gamma, max_depth=args.max_depth, verbose=args.verbose,
                       random_state=args.rs)
        detrace = detrace.fit(X_train, y_train)
        logger.info('{:.3f}s'.format(time.time() - start))

        proba = detrace.predict_proba(X_test)[:, 1]
        pred = detrace.predict(X_test)
        aucs.append(roc_auc_score(y_test, proba))
        accs.append(accuracy_score(y_test, pred))

        exp_util.performance(detrace, X_test, y_test, name='DeTRACE', logger=logger)

    if args.save_results:
        np.save(os.path.join(out_dir, 'auc.npy'), aucs)
        np.save(os.path.join(out_dir, 'acc.npy'), accs)
        np.save(os.path.join(out_dir, 'gamma.npy'), gammas)


def main(args):

    # run experiment multiple times
    for i in range(args.repeats):

        # create output dir
        ep_dir = os.path.join(args.out_dir, args.dataset, 'rs{}'.format(args.rs), 'ep{}'.format(args.epsilon))
        os.makedirs(ep_dir, exist_ok=True)

        # create logger
        logger = print_util.get_logger(os.path.join(ep_dir, 'log.txt'))
        logger.info(args)
        logger.info('\nRun {}, seed: {}'.format(i + 1, args.rs))

        # run experiment
        vary_gamma(args, logger, ep_dir, seed=args.rs)
        args.rs += 1

        # remove logger
        print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='output/epsilon_gamma', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='synthetic', help='dataset to use for the experiment.')
    parser.add_argument('--n_samples', type=int, default=10, help='number of samples to generate.')
    parser.add_argument('--n_attributes', type=int, default=4, help='number of attributes to generate.')
    parser.add_argument('--test_frac', type=float, default=0.2, help='fraction of data to use for testing.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--repeats', type=int, default=1, help='number of times to perform the experiment.')
    parser.add_argument('--save_results', action='store_true', default=False, help='save results.')
    parser.add_argument('--epsilon', type=float, default=0.1, help='efficiency parameter for tree.')
    parser.add_argument('--max_depth', type=int, default=4, help='maximum depth of the tree.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    main(args)
