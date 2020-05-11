"""
Print results of amortize experiment for a single dataset.
"""
import os
import argparse
from datetime import datetime

import numpy as np

import print_util


def print_dataset(args, dataset, logger):

    for adversary in args.adversary:
        logger.info('\nDataset: {}, Adversary: {}'.format(dataset, adversary))

        naive = print_util.get_results(args, dataset, adversary, 'naive')
        exact = print_util.get_results(args, dataset, adversary, 'exact')

        n_train, _ = print_util.get_mean1d(args, naive, 'n_train', as_int=True)
        n_features, _ = print_util.get_mean1d(args, naive, 'n_features', as_int=True)
        n_trees, _ = print_util.get_mean1d(args, naive, 'n_estimators', as_int=True)
        max_depth, _ = print_util.get_mean1d(args, naive, 'max_depth', as_int=True)
        max_features = print_util.get_max_features(args, naive, 'max_features')

        s = '{:,} instances   {:,} features   Trees: {:,}   '
        s += 'Max depth: {}   Max features: {}'
        logger.info(s.format(n_train, n_features, n_trees, max_depth, max_features))

        naive_train, _ = print_util.get_mean1d(args, naive, 'train_time')
        naive_amortize, _ = print_util.get_mean_amortize(args, naive)
        s = '[Naive] train time: {:.3f}s, amortized: {:.5f}s'
        logger.info(s.format(naive_train, naive_amortize))

        exact_train, _ = print_util.get_mean1d(args, exact, 'train_time')
        exact_amortize, _ = print_util.get_mean_amortize(args, exact)
        exact_retrains, _ = print_util.get_mean_retrainings(args, exact)
        exact_speedup = int(naive_amortize / exact_amortize)
        exact_scores, _ = print_util.get_mean(args, exact, args.metric)
        exact_deletions, _ = print_util.get_mean_completions(args, exact, n_trees)
        exact_depth, _ = print_util.get_mean_retrain_depth(args, exact)
        exact_n_scores = len(exact_scores)
        s = '[Exact] train: {:.3f}s, completed: {:,}, '
        s += 'amortized: {:.5f}s, speedup: {:>7}x, retrains: {:7,}, retrain depth: {}'
        logger.info(s.format(exact_train, exact_deletions, exact_amortize,
                    exact_speedup, exact_retrains, exact_depth))

        for epsilon in args.epsilon:
            cedar = print_util.get_results(args, dataset, adversary, 'cedar_ep{}'.format(epsilon))

            lmbda, _ = print_util.get_mean1d(args, cedar, 'lmbda', as_int=True)
            cedar_train, _ = print_util.get_mean1d(args, cedar, 'train_time')
            cedar_amortize, _ = print_util.get_mean_amortize(args, cedar)
            cedar_retrains, _ = print_util.get_mean_retrainings(args, cedar)
            cedar_speedup = int(naive_amortize / cedar_amortize)
            cedar_scores, _ = print_util.get_mean(args, cedar, args.metric)
            cedar_deletions, _ = print_util.get_mean_completions(args, cedar, n_trees)
            cedar_depth, _ = print_util.get_mean_retrain_depth(args, cedar)
            cedar_n_scores = len(cedar_scores)
            n_scores = min(exact_n_scores, cedar_n_scores)
            score_diff = np.abs(exact_scores[:n_scores] - cedar_scores[:n_scores]).mean()
            s = '[CeDAR] train: {:.3f}s, completed: {:,}, amortized: {:.5f}s, '
            s += 'speedup: {:>7}x, retrains: {:7,}, retrain depth: {}, '
            s += 'epsilon: {:>4}, lmbda: {}, {} diff: {:.5f}'
            logger.info(s.format(cedar_train, cedar_deletions, cedar_amortize, cedar_speedup,
                        cedar_retrains, cedar_depth,
                        epsilon, lmbda, args.metric, score_diff))


def main(args):

    # create logger
    os.makedirs(args.out_dir, exist_ok=True)
    logger_name = 'amortize.txt'
    logger = print_util.get_logger(os.path.join(args.out_dir, logger_name))
    logger.info(args)
    logger.info(datetime.now())

    for dataset in args.dataset:
        print_dataset(args, dataset, logger)
        logger.info('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='output/amortize/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/prints/', help='output directory.')
    parser.add_argument('--dataset', type=str, nargs='+', default=['surgical'], help='dataset to print.')
    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='random state.')
    parser.add_argument('--repeats', type=int, default=5, help='number of experiments.')
    parser.add_argument('--metric', type=str, default='auc', help='predictive performance metric.')
    parser.add_argument('--model_type', type=str, default='forest', help='stump, tree, or forest.')
    parser.add_argument('--epsilon', type=str, nargs='+', default=['0.1', '0.25', '0.5', '1.0'], help='epsilon.')
    parser.add_argument('--adversary', type=str, nargs='+', default=['random', 'root'], help='adversary to show.')
    parser.add_argument('--criterion', type=str, default='gini', help='splitting criterion.')
    args = parser.parse_args()
    main(args)
