"""
Print delete_until_retrain results.
"""
import os
import argparse
from datetime import datetime

import numpy as np

import print_util


def main(args):

    # create logger
    logger_name = 'deletions.txt'
    logger = print_util.get_logger(os.path.join(args.out_dir, logger_name))
    logger.info(args)
    logger.info(datetime.now())

    for model_type in args.model_type:
        print('\n{}'.format(model_type.capitalize()))

        for adversary in args.adversary:
            print('\n{}'.format(adversary.capitalize()))

            for dataset in args.dataset:

                # get results
                r = {}
                for rs in args.rs:
                    fp = os.path.join(args.in_dir, dataset, model_type, args.criterion, adversary,
                                      'rs{}'.format(rs), 'results.npy')
                    r[rs] = np.load(fp, allow_pickle=True)[()]

                n_remove_mean, _ = print_util.get_mean(args, r, 'n_deletions')
                train_time_mean, _ = print_util.get_mean1d(args, r, name='train_time')
                epsilons = r[args.rs]['epsilon']

                n_train = r[args.rs]['n_train']
                n_features = r[args.rs]['n_features']

                out_str = '\n{} ({:,} instances, {:,} features), train_time: {:.5f}s'
                print(out_str.format(dataset, n_train, n_features, train_time_mean))

                for i, epsilon in enumerate(epsilons):
                    print('epsilon: {:7,} => n_remove: {:,}'.format(epsilon, int(round(n_remove_mean[i]))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='output/delete_until_retrain/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/prints/', help='output directory.')
    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='initial seed.')
    parser.add_argument('--dataset', type=str, nargs='+', default=['mfc19'], help='datasets to show.')
    parser.add_argument('--adversary', type=str, nargs='+', default=['random'], help='adversary to show.')
    parser.add_argument('--model_type', type=str, nargs='+', default=['stump'], help='models to show.')
    parser.add_argument('--criterion', type=str, default='gini', help='splitting criterion.')
    args = parser.parse_args()
    main(args)
