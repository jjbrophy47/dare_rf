"""
Plot results of no_retrain experiment.
"""
import os
import argparse
from datetime import datetime

import numpy as np

import print_util


def main(args):

    # create logger
    os.makedirs(args.out_dir, exist_ok=True)
    logger_name = 'no_retrain.txt'
    logger = print_util.get_logger(os.path.join(args.out_dir, logger_name))
    logger.info(args)
    logger.info(datetime.now())

    for model_type in args.model_type:
        print('\n{}'.format(model_type.capitalize()))

        for dataset in args.dataset:

            # get results
            r = {}
            for rs in args.rs:
                fp = os.path.join(args.in_dir, dataset, model_type,
                                  args.criterion,
                                  'rs{}'.format(rs), 'results.npy')
                r[rs] = np.load(fp, allow_pickle=True)[()]

            lmbda, _, lmbdas = print_util.get_mean(args, r, key='lmbda')
            n_train = r[args.rs]['n_train']
            n_features = r[args.rs]['n_features']
            max_depth = r[args.rs]['max_depth']
            n_estimators = r[args.rs].get('n_estimators')
            max_features = r[args.rs].get('max_features')
            lmbda_step_size = r[args.rs].get('lmbda_step_size')

            n_trees = 1 if n_estimators is None else n_estimators
            adjusted_lmbdas = [lm / 5 / max_depth / n_trees for lm in lmbdas]

            out_str = '\n{} ({:,} instances, {:,} features), depth: {}, trees: {}, features: {}'
            out_str += ', lmbda_step_size: {}, lmbda: {}'
            print(out_str.format(dataset, n_train, n_features, max_depth, n_estimators,
                                 max_features, lmbda_step_size, lmbda))
            print('lmbdas: {}'.format(lmbdas))
            print('adjusted lmbdas: {}'.format(adjusted_lmbdas))

            gammas = [1 / n_train, 0.001]  # removing 1 and 0.1%
            n_remove_label = ['1', '0.1%']
            for i, gamma in enumerate(gammas):
                epsilon = lmbda * gamma
                print('n_remove: {}\tepsilon: {:9.5f}'.format(n_remove_label[i], epsilon))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='output/no_retrain/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/prints/', help='output directory.')
    parser.add_argument('--rs', type=int, default=1, help='initial seed.')
    parser.add_argument('--repeats', type=int, default=5, help='number of repeated results to include.')
    parser.add_argument('--dataset', type=str, nargs='+', default=['mfc20'], help='datasets to show.')
    parser.add_argument('--model_type', type=str, nargs='+', default=['forest'], help='models to show.')
    parser.add_argument('--criterion', type=str, default='gini', help='splitting criterion.')
    args = parser.parse_args()
    main(args)
