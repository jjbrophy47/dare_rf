"""
Experiment: How much data can we delete before we need to retrain?
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
from utility import data_util, exp_util, print_util, exact_adv_util, cert_adv_util


def first_retrain(args, logger, out_dir, seed):

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, seed, data_dir=args.data_dir)

    # choose instances to delete
    n_remove = X_train.shape[0]

    if args.adversary == 'random':
        np.random.seed(seed)
        delete_indices = np.random.choice(X_train.shape[0], size=n_remove, replace=False)

    elif args.adversary == 'root':
        delete_indices = exact_adv_util.exact_adversary(X_train, y_train, n_samples=n_remove, seed=seed,
                                                        verbose=args.verbose, logger=logger)
    elif args.adversary == 'certified':
        delete_indices = cert_adv_util.certified_adversary(X_train, y_train, epsilon=args.epsilon, lmbda=args.lmbda,
                                                           n_samples=n_remove, seed=seed,
                                                           verbose=args.verbose, logger=logger)
    else:
        exit('uknown adversary: {}'.format(args.adversary))

    logger.info('num delete instances: {:,}'.format(len(delete_indices)))
    logger.info('adversary: {}'.format(args.adversary))

    # dataset statistics
    logger.info('num train instances: {:,}'.format(X_train.shape[0]))
    logger.info('num test instances: {:,}'.format(X_test.shape[0]))
    logger.info('num features: {:,}'.format(X_train.shape[1]))

    logger.info('\nExact')
    start = time.time()
    model = cedar.Tree(lmbda=-1, max_depth=args.max_depth, verbose=args.verbose, random_state=seed)
    model = model.fit(X_train, y_train)
    logger.info('{:.3f}s'.format(time.time() - start))

    exp_util.performance(model, X_test, y_test, name='exact', logger=logger)

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

    epsilons = [0.01, 0.1, 1.0]
    lmbdas = np.linspace(args.min_lmbda, args.max_lmbda, args.n_lmbda)

    logger.info('n_remove: {:,}'.format(n_remove))
    logger.info('epsilons: {}'.format(epsilons))

    random_state = exp_util.get_random_state(seed)

    n_deletions_list = []
    aucs_list = []
    accs_list = []

    for i, epsilon in enumerate(epsilons):
        logger.info('\nepsilon: {:.0e}'.format(epsilon))
        n_deletions = []
        aucs = []
        accs = []

        for j, lmbda in enumerate(lmbdas):

            start = time.time()
            model = cedar.Tree(epsilon=epsilon, lmbda=lmbda, max_depth=args.max_depth,
                               verbose=args.verbose, random_state=random_state)
            model = model.fit(X_train, y_train)

            proba = model.predict_proba(X_test)[:, 1]
            pred = model.predict(X_test)
            auc = roc_auc_score(y_test, proba)
            acc = accuracy_score(y_test, pred)

            for k, delete_ndx in enumerate(delete_indices):
                model.delete(delete_ndx)
                delete_types, delete_depths = model.get_removal_statistics()

                if delete_types[0] == 2:
                    break

            out_str = '[{}] lmbda: {:.2e} => deletions: {:,}, acc: {:.3f}, auc: {:.3f}'
            logger.info(out_str.format(j, lmbda, k, acc, auc))

            n_deletions.append(k)
            aucs.append(auc)
            accs.append(acc)

        n_deletions_list.append(n_deletions)
        aucs_list.append(aucs)
        accs_list.append(accs)
    n_deletions_arr = np.vstack(n_deletions_list)
    aucs_arr = np.vstack(aucs_list)
    accs_arr = np.vstack(accs_list)

    if args.save_results:
        d = model.get_params()
        d['n_deletions'] = n_deletions_arr
        d['auc'] = aucs_arr
        d['acc'] = accs_arr
        d['epsilon'] = epsilons
        d['lmbda'] = lmbdas
        d['n_train'] = X_train.shape[0]
        d['n_features'] = X_train.shape[1]
        d['n_remove'] = n_remove
        d['adversary'] = args.adversary
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
        first_retrain(args, logger, ep_dir, seed=args.rs)
        args.rs += 1

        # remove logger
        print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='output/tree/first_retrain', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='synthetic', help='dataset to use for the experiment.')

    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--repeats', type=int, default=1, help='number of times to perform the experiment.')
    parser.add_argument('--save_results', action='store_true', default=False, help='save results.')

    parser.add_argument('--min_lmbda', type=float, default=0, help='minimum lambda.')
    parser.add_argument('--max_lmbda', type=float, default=2, help='maximum lambda.')
    parser.add_argument('--n_lmbda', type=int, default=10, help='number of data points.')

    parser.add_argument('--adversary', type=str, default='random', help='type of adversarial ordering.')

    parser.add_argument('--max_depth', type=int, default=1, help='maximum depth of the tree.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    args = parser.parse_args()
    main(args)
