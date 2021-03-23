"""
Organize the performance results into a single csv.
"""
import os
import sys
import argparse
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import sem
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from utility import print_util


def get_result(template, in_dir):
    """
    Obtain the results for this baseline method.
    """
    result = template.copy()

    fp = os.path.join(in_dir, 'results.npy')

    if not os.path.exists(fp):
        result = None

    else:
        d = np.load(fp, allow_pickle=True)[()]
        result.update(d)

    return result


def process_utility(gf, tol_list=[0.0, 0.1, 0.25, 0.5, 1.0]):
    """
    Processes utility differences BEFORE addition/deletion,
    and averages the results over different random states.
    """
    result = {}

    result['sklearn_mem_mean'] = np.mean(gf['sklearn_mem'])
    result['sklearn_mem_sem'] = sem(gf['sklearn_mem'])

    result['sklearn_train_mean'] = np.mean(gf['sklearn_train'])
    result['sklearn_train_std'] = sem(gf['sklearn_train'])

    for tol in tol_list:
        model = 'dare_{}'.format(tol)

        result['{}_mem_mean'.format(model)] = np.mean(gf['{}_mem'.format(model)])
        result['{}_mem_sem'.format(model)] = sem(gf['{}_mem'.format(model)])

        result['{}_train_mean'.format(model)] = np.mean(gf['{}_train'.format(model)])
        result['{}_train_std'.format(model)] = np.std(gf['{}_train'.format(model)])

    return result


def process_results(df):
    """
    Averages utility results over different random states.
    """

    groups = ['dataset', 'criterion']

    main_result_list = []

    for tup, gf in tqdm(df.groupby(groups)):
        main_result = {k: v for k, v in zip(groups, tup)}
        main_result.update(process_utility(gf))
        main_result['n_estimators'] = gf['n_estimators'].mode()[0]
        main_result['max_depth'] = gf['max_depth'].mode()[0]
        main_result['num_runs'] = len(gf)
        main_result_list.append(main_result)

    main_df = pd.DataFrame(main_result_list)

    return main_df


def create_csv(args, out_dir, logger):

    logger.info('\nGathering results...')

    experiment_settings = list(product(*[args.dataset, args.criterion, args.rs]))

    results = []
    for dataset, criterion, rs in tqdm(experiment_settings):
        template = {'dataset': dataset, 'criterion': criterion, 'rs': rs}
        experiment_dir = os.path.join(args.in_dir,
                                      dataset,
                                      criterion,
                                      'rs_{}'.format(rs))

        # skip empty experiments
        if not os.path.exists(experiment_dir):
            continue

        # add results to result dict
        result = get_result(template, experiment_dir)
        if result is not None:
            results.append(result)

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 180)

    df = pd.DataFrame(results)
    logger.info('\nRaw results:\n{}'.format(df))

    logger.info('\nProcessing results...')
    main_df = process_results(df)
    logger.info('\nProcessed results:\n{}'.format(main_df))

    main_df.to_csv(os.path.join(out_dir, 'results.csv'), index=None)


def main(args):

    out_dir = os.path.join(args.out_dir)

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    create_csv(args, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--in_dir', type=str, default='output/memory/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/memory/csv/', help='output directory.')

    # experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['surgical', 'vaccine', 'adult', 'bank_marketing', 'flight_delays', 'diabetes',
                                 'no_show', 'olympics', 'census', 'credit_card', 'twitter', 'synthetic',
                                 'higgs', 'ctr'], help='dataset.')
    parser.add_argument('--criterion', type=str, nargs='+', default=['gini', 'entropy'], help='criterion.')
    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='random state.')

    args = parser.parse_args()
    main(args)
