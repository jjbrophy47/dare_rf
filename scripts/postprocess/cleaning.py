"""
Organize results into a single CSV.
"""
import os
import sys
import argparse
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import sem

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from utility import print_util


def get_result(template, in_dir):
    """
    Obtain results.
    """
    result = template.copy()

    fp = os.path.join(in_dir, 'results.npy')

    if not os.path.exists(fp):
        result = None

    else:
        d = np.load(fp, allow_pickle=True)[()]
        result.update(d)

    return result


def process_results(df):
    """
    Average results over different random states.
    """
    groups = ['dataset', 'criterion', 'method']

    main_result_list = []

    for tup, gf in tqdm(df.groupby(groups)):
        main_result = {k: v for k, v in zip(groups, tup)}
        main_result['auc_clean'] = gf['auc_clean'].mean()
        main_result['acc_clean'] = gf['acc_clean'].mean()
        main_result['ap_clean'] = gf['ap_clean'].mean()
        main_result['num_runs'] = len(gf)

        # aggregate list results
        auc_list = []
        acc_list = []
        ap_list = []
        checked_pct_list = []
        for row in gf.itertuples(index=False):
            auc_list.append(list(row.auc))
            acc_list.append(list(row.acc))
            ap_list.append(list(row.ap))
            checked_pct_list.append(list(row.checked_pct))

        main_result['auc'] = np.mean(auc_list, axis=0)
        main_result['acc'] = np.mean(acc_list, axis=0)
        main_result['ap'] = np.mean(ap_list, axis=0)

        main_result['auc_std'] = sem(auc_list, axis=0)
        main_result['acc_std'] = sem(acc_list, axis=0)
        main_result['ap_std'] = sem(ap_list, axis=0)

        main_result['checked_pct'] = np.mean(checked_pct_list, axis=0)

        main_result_list.append(main_result)

    main_df = pd.DataFrame(main_result_list)

    return main_df


def create_csv(args, out_dir, logger):

    logger.info('\nGathering results...')

    experiment_settings = list(product(*[args.dataset, args.criterion, args.method, args.rs]))

    results = []
    for dataset, criterion, method, rs in tqdm(experiment_settings):
        template = {'dataset': dataset, 'criterion': criterion, 'method': method, 'rs': rs}
        experiment_dir = os.path.join(args.in_dir, dataset, criterion, method, 'rs_{}'.format(rs))

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
    parser.add_argument('--in_dir', type=str, default='output/cleaning/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/cleaning/csv/', help='output directory.')

    # experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['surgical', 'vaccine', 'adult', 'bank_marketing', 'flight_delays', 'diabetes',
                                 'olympics', 'census', 'credit_card', 'synthetic', 'higgs'], help='dataset.')
    parser.add_argument('--criterion', type=str, nargs='+', default=['gini', 'entropy'], help='criterion.')
    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='random state.')
    parser.add_argument('--method', type=str, nargs='+', default=['random', 'dart', 'dart_loss'], help='method.')

    args = parser.parse_args()
    main(args)
