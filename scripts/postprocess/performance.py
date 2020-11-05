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


def process_utility(gf):
    """
    Processes utility differences BEFORE addition/deletion,
    and averages the results over different random states.
    """
    result = {}

    acc_list = []
    auc_list = []
    ap_list = []

    for row in gf.itertuples(index=False):
        acc_list.append(row.acc)
        auc_list.append(row.auc)
        ap_list.append(row.ap)

    result['acc_mean'] = np.mean(acc_list)
    result['acc_std'] = sem(acc_list)

    result['auc_mean'] = np.mean(auc_list)
    result['auc_std'] = sem(auc_list)

    result['ap_mean'] = np.mean(ap_list)
    result['ap_std'] = sem(ap_list)

    return result


def process_results(df):
    """
    Averages utility results over different random states.
    """

    df['continuous'] = df['continuous'].fillna(False)

    groups = ['dataset', 'criterion', 'model', 'bootstrap', 'continuous']

    main_result_list = []

    df['max_features'] = df['max_features'].fillna(-1)

    for tup, gf in tqdm(df.groupby(groups)):
        main_result = {k: v for k, v in zip(groups, tup)}
        main_result.update(process_utility(gf))
        main_result['n_estimators'] = gf['n_estimators'].mode()[0]
        main_result['max_depth'] = gf['max_depth'].mode()[0]
        main_result['max_features'] = gf['max_features'].mode()[0]
        main_result['num_runs'] = len(gf)
        main_result_list.append(main_result)

    main_df = pd.DataFrame(main_result_list)

    return main_df


def create_csv(args, out_dir, logger):

    logger.info('\nGathering results...')

    experiment_settings = list(product(*[args.dataset, args.criterion, args.rs, args.model, args.tuning]))

    results = []
    for dataset, criterion, rs, model, tuning in tqdm(experiment_settings):
        template = {'dataset': dataset, 'criterion': criterion, 'rs': rs, 'model': model}
        experiment_dir = os.path.join(args.in_dir, dataset, criterion,
                                      tuning, 'rs_{}'.format(rs), model)

        # skip empty experiments
        if not os.path.exists(experiment_dir):
            continue

        # add results to result dict
        result = get_result(template, experiment_dir)
        if result is not None:
            results.append(result)

        # add sklearn results
        if model == 'sklearn':

            # add bootstrap result
            bootstrap_dir = os.path.join(experiment_dir, 'bootstrap')
            bootstrap_result = get_result(template, bootstrap_dir)
            if bootstrap_result is not None:
                results.append(bootstrap_result)

            # add continuous result
            continuous_dir = os.path.join(args.in_dir, dataset, criterion,
                                          'continuous', tuning,
                                          'rs_{}'.format(rs), model)
            continuous_result = get_result(template, continuous_dir)
            if continuous_result is not None:
                results.append(continuous_result)

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 180)

    df = pd.DataFrame(results)
    logger.info('\nRaw results:\n{}'.format(df))

    logger.info('\nProcessing results...')
    main_df = process_results(df)
    logger.info('\nProcessed results:\n{}'.format(main_df))

    # uncomment to see performance differences between continuous and binary feature transformations
    print(main_df[(main_df['model'] == 'sklearn') & (main_df['bootstrap'] == False)])

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
    parser.add_argument('--in_dir', type=str, default='output/performance/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/performance/csv/', help='output directory.')

    # experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['surgical', 'vaccine', 'adult', 'bank_marketing', 'flight_delays', 'diabetes',
                                 'olympics', 'skin', 'census', 'credit_card', 'twitter', 'gas_sensor',
                                 'synthetic', 'higgs'], help='dataset.')
    parser.add_argument('--criterion', type=str, nargs='+', default=['gini', 'entropy'], help='criterion.')
    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='random state.')
    parser.add_argument('--model', type=int, nargs='+', default=['exact', 'random', 'sklearn', 'borat'], help='model.')
    parser.add_argument('--tuning', type=int, nargs='+', default=['tuned', 'no_tune'], help='tuning option.')

    args = parser.parse_args()
    main(args)
