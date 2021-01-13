"""
Analyzes how many unique features there are for each dataset.
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
from utility import data_util
from utility import print_util


def extract_statistics(dataset, X_train, logger):
    """
    Extract feature statistics, e.g. no. features,
    no. unique values per feature, no. candidate
    threhsolds per feature, etc.
    """
    logger.info('\n{}'.format(dataset.capitalize()))
    logger.info('no. samples: {:,}'.format(X_train.shape[0]))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    num_unary_features = 0
    num_binary_features = 0
    nonbinary_vals = []

    for i in range(X_train.shape[1]):

        num_unique_vals = len(np.unique(X_train[:, i]))
        assert num_unique_vals > 0

        if num_unique_vals == 1:
            num_unary_features += 1

        elif num_unique_vals == 2:
            num_binary_features += 1

        else:
            nonbinary_vals.append(num_unique_vals)

    logger.info('no. unary features: {:,}'.format(num_unary_features))
    logger.info('no. binary features: {:,}'.format(num_binary_features))

    if len(nonbinary_vals) > 0:

        nb_min = np.min(nonbinary_vals)
        nb_max = np.max(nonbinary_vals)
        nb_mean = np.mean(nonbinary_vals)
        nb_median = np.median(nonbinary_vals)

        s = '    [values] min.: {}, max.: {}, mean: {}, median: {}'
        logger.info('no. nonbinary features: {:,}'.format(len(nonbinary_vals)))
        logger.info(s.format(nb_min, nb_max, nb_mean, nb_median))

    result = {}
    result['num_features'] = X_train.shape[1]

    return result


def experiment(args, logger, out_dir):
    """
    Extracts feature statistics from each dataset.
    """
    results = []
    for dataset in args.dataset:
        X_train, _, y_train, _ = data_util.get_data(dataset,
                                                    data_dir=args.data_dir,
                                                    continuous=True)
        result = extract_statistics(dataset, X_train, logger)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, 'results.csv'), index=None)


def main(args):

    log_fp = os.path.join(args.out_dir, 'log.txt')
    os.makedirs(args.out_dir, exist_ok=True)

    # create logger
    logger = print_util.get_logger(log_fp)
    logger.info(args)
    logger.info(datetime.now())

    # run experiment
    experiment(args, logger, args.out_dir)

    # remove logger
    print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--out_dir', type=str, default='output/continuous/', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', type=str, nargs='+', help='dataset to use for the experiment.',
                        default=['vaccine', 'surgical', 'adult', 'bank_marketing', 'census', 'credit_card',
                                 'census', 'diabetes', 'flight_delays', 'olympics', 'synthetic', 'higgs'])
    args = parser.parse_args()
    main(args)
