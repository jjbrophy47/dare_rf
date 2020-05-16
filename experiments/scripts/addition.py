"""
Experiment: Compute the amortized runtime of addtions.
"""
import os
import sys
import time
import argparse
from collections import Counter
from datetime import datetime

import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import cedar
from utility import data_util
from utility import exp_util
from utility import print_util
from utility import root_adversary


def _get_model(args, epsilon=0, lmbda=-1, random_state=None):
    """
    Return the appropriate model CeDAR model.
    """

    if args.model_type == 'stump':
        model = cedar.Tree(epsilon=epsilon,
                           lmbda=lmbda,
                           max_depth=1,
                           criterion=args.criterion,
                           verbose=args.verbose,
                           random_state=random_state)

    elif args.model_type == 'tree':
        model = cedar.Tree(epsilon=epsilon,
                           lmbda=lmbda,
                           max_depth=args.max_depth,
                           criterion=args.criterion,
                           verbose=args.verbose,
                           random_state=random_state)

    elif args.model_type == 'forest':
        max_features = None if args.max_features == -1 else args.max_features
        model = cedar.Forest(epsilon=epsilon,
                             lmbda=lmbda,
                             max_depth=args.max_depth,
                             criterion=args.criterion,
                             n_estimators=args.n_estimators,
                             max_features=max_features,
                             verbose=args.verbose,
                             random_state=random_state)

    else:
        exit('model_type {} unknown!'.format(args.model_type))

    return model


def learning_method(args, lmbda, random_state, out_dir, logger,
                    X_train, y_train, X_add, y_add, X_test, y_test, name):

    experiment_start = time.time()
    n_add = X_add.shape[0]

    logger.info('\n{}'.format(name.capitalize()))
    logger.info('experiment start: {}'.format(datetime.now()))

    # train
    epsilon = 0 if name == 'exact' else args.epsilon
    lmbda = -1 if name == 'exact' else lmbda
    logger.info('lmbda: {}'.format(lmbda))
    model = _get_model(args, epsilon=epsilon, lmbda=lmbda, random_state=random_state)

    start = time.time()
    model = model.fit(X_train, y_train)
    train_time = time.time() - start
    logger.info('[{}] train time: {:.3f}s'.format(name, train_time))
    auc, acc = exp_util.performance(model, X_test, y_test, logger=logger, name=name)

    # result containers
    performance_markers = [int(x) for x in np.linspace(0, n_add - 1, 10)]
    times = [train_time]
    aucs, accs = [auc], [acc]

    # add as many indices as possible while time allows
    i = 0
    while time.time() - experiment_start < args.time_limit and i < n_add:

        start = time.time()
        model.add(X_add[[i]], y_add[[i]])
        end_time = time.time() - start
        times.append(end_time)

        if i in performance_markers:
            auc, acc = exp_util.performance(model, X_test, y_test, logger=None, name=name)
            aucs.append(auc)
            accs.append(acc)

            if args.verbose > 0:
                experiment_time = time.time() - experiment_start
                percent_complete = i / n_add * 100
                status_str = '[{:.2f}%] experiment time: {:.3f}s, amortized time: {:.3f}s'
                logger.info(status_str.format(percent_complete, experiment_time, np.mean(times)))

        i += 1

    types, depths = model.get_add_statistics()
    types_counter = Counter(types)
    depths_counter = Counter(depths)
    logger.info('[{}] completed additions: {:,}'.format(name, i))
    logger.info('[{}] amortized: {:.7f}s'.format(name, np.mean(times)))
    logger.info('[{}] types: {}'.format(name, types_counter))
    logger.info('[{}] depths: {}'.format(name, depths_counter))
    auc, acc = exp_util.performance(model, X_test, y_test, logger=logger, name=name)
    logger.info('[{}] total time: {:.3f}s'.format(name, time.time() - experiment_start))

    if args.save_results:
        d = model.get_params()
        d['train_time'] = train_time
        d['time'] = np.array(times)
        d['type'] = np.array(types)
        d['depth'] = np.array(depths)
        d['auc'] = np.array(aucs)
        d['acc'] = np.array(accs)

        fname = name if name == 'exact' else '{}_ep{}'.format(name, args.epsilon)
        np.save(os.path.join(out_dir, '{}.npy'.format(fname)), d)


def naive_method(args, random_state, out_dir, logger, X_train, y_train,
                 X_add, y_add, X_test, y_test, name):

    n_add = X_add.shape[0]

    # train
    logger.info('\n{}'.format(name.capitalize()))
    model = _get_model(args, epsilon=0, lmbda=-1, random_state=random_state)

    start = time.time()
    model = model.fit(X_train, y_train)
    initial_train_time = time.time() - start
    logger.info('[{}] train time: {:.3f}s'.format(name, initial_train_time))
    auc, acc = exp_util.performance(model, X_test, y_test, logger=logger, name=name)

    # train again after adding all instances and draw a line between the two points
    X_train_new = np.vstack([X_train, X_add])
    y_train_new = np.concatenate([y_train, y_add])

    start = time.time()
    model = _get_model(args, epsilon=0, lmbda=-1, random_state=random_state)
    model = model.fit(X_train_new, y_train_new)
    last_train_time = time.time() - start
    times = np.linspace(initial_train_time, last_train_time, n_add - 1)

    logger.info('[{}] amortized: {:.3f}s'.format(name, np.mean(times)))
    exp_util.performance(model, X_test, y_test, logger=logger, name=name)

    if args.save_results:
        d = model.get_params()
        d['train_time'] = initial_train_time
        d['time'] = np.array(times)
        d['n_train'] = X_train.shape[0]
        d['n_features'] = X_train.shape[1]
        np.save(os.path.join(out_dir, 'naive.npy'), d)


def experiment(args, logger, out_dir, seed, lmbda):

    # obtain data
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, data_dir=args.data_dir)

    # dataset statistics
    logger.info('train instances: {:,}'.format(X_train.shape[0]))
    logger.info('test instances: {:,}'.format(X_test.shape[0]))
    logger.info('features: {:,}'.format(X_train.shape[1]))
    logger.info('split criterion: {}'.format(args.criterion))

    random_state = exp_util.get_random_state(seed)

    # choose instances to add
    n_add = args.n_add if args.frac_add is None else int(X_train.shape[0] * args.frac_add)

    # order the samples based on an adversary
    if args.adversary == 'random':
        np.random.seed(random_state)
        add_indices = np.random.choice(X_train.shape[0], size=n_add, replace=False)

    elif args.adversary == 'root':
        add_indices = root_adversary.order_samples(X_train, y_train, n_samples=n_add,
                                                   criterion=args.criterion,
                                                   seed=random_state, verbose=args.verbose,
                                                   logger=logger)

    else:
        exit('unknown adversary: {}'.format(args.adversary))

    # separate add samples from the rest of the training data
    add_order = add_indices[::-1]
    X_add = X_train[add_order]
    y_add = y_train[add_order]

    train_indices = np.setdiff1d(np.arange(X_train.shape[0]), add_indices)
    X_train_sub = X_train[train_indices]
    y_train_sub = y_train[train_indices]

    logger.info('instances to add: {:,}'.format(len(add_indices)))
    logger.info('adversary: {}'.format(args.adversary))

    # naive retraining method
    if args.naive:
        naive_method(args, random_state, out_dir, logger, X_train_sub, y_train_sub,
                     X_add, y_add, X_test, y_test, 'naive')

    # exact unlearning method
    if args.exact:
        learning_method(args, lmbda, random_state, out_dir, logger, X_train_sub, y_train_sub,
                        X_add, y_add, X_test, y_test, 'exact')

    # approximate unlearning method
    if args.cedar:
        learning_method(args, lmbda, random_state, out_dir, logger, X_train_sub, y_train_sub,
                        X_add, y_add, X_test, y_test, 'cedar')


def main(args):

    # create output dir
    rs_dir = os.path.join(args.out_dir, args.dataset, args.model_type,
                          args.criterion, args.adversary,
                          'rs{}'.format(args.rs))
    os.makedirs(rs_dir, exist_ok=True)

    # create logger
    logger_name = 'log_ep{}.txt'.format(args.epsilon) if args.cedar else 'log.txt'
    logger = print_util.get_logger(os.path.join(rs_dir, logger_name))
    logger.info(args)
    logger.info(datetime.now())
    logger.info('\nSeed: {}, Advesary: {}'.format(args.rs, args.adversary))

    # run experiment
    experiment(args, logger, rs_dir, seed=args.rs, lmbda=args.lmbda)

    # remove logger
    print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment settings
    parser.add_argument('--out_dir', type=str, default='output/addition/', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='surgical', help='dataset to use for the experiment.')
    parser.add_argument('--model_type', type=str, default='forest', help='stump, tree, or forest.')
    parser.add_argument('--rs', type=int, default=1, help='seed to enhance reproducibility.')
    parser.add_argument('--save_results', action='store_true', default=True, help='save results.')
    parser.add_argument('--time_limit', type=int, default=86400, help='maximum number of seconds.')

    # methods
    parser.add_argument('--naive', action='store_true', default=False, help='Include retrain baseline.')
    parser.add_argument('--exact', action='store_true', default=False, help='Include deterministic baseline.')
    parser.add_argument('--cedar', action='store_true', default=False, help='Include cedar model.')

    # model hyperparameters
    parser.add_argument('--epsilon', type=float, default=1.0, help='setting for certified adversarial ordering.')
    parser.add_argument('--lmbda', type=float, default=0, help='noise hyperparameter.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in the forest.')
    parser.add_argument('--max_features', type=float, default=-1, help='maximum features to sample.')
    parser.add_argument('--max_depth', type=int, default=1, help='maximum depth of the tree.')
    parser.add_argument('--criterion', type=str, default='gini', help='splitting criterion.')

    # adversary settings
    parser.add_argument('--n_add', type=int, default=10, help='number of instances to sequentially add.')
    parser.add_argument('--frac_add', type=float, default=0.1, help='fraction of instances to add.')
    parser.add_argument('--adversary', type=str, default='random', help='type of adversarial ordering.')

    # display settings
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level.')

    args = parser.parse_args()
    main(args)
