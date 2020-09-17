"""
Organize the worst-case adversary results into a single csv.
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


def process_periodic_utility(gf):
    """
    Processes utility differences BEFORE and DURING addition/deletion,
    and averages the results over different random states.
    """
    result = {}

    acc_mean_list, acc_std_list = [], []
    auc_mean_list, auc_std_list = [], []
    ap_mean_list, ap_std_list = [], []

    acc_diff_mean_list, acc_diff_std_list = [], []
    auc_diff_mean_list, auc_diff_std_list = [], []
    ap_diff_mean_list, ap_diff_std_list = [], []

    for row in gf.itertuples(index=False):

        acc_mean_list.append(np.mean(row.acc))
        auc_mean_list.append(np.mean(row.auc))
        ap_mean_list.append(np.mean(row.ap))

        acc_std_list.append(np.std(row.acc))
        auc_std_list.append(np.std(row.auc))
        ap_std_list.append(np.std(row.ap))

        acc_diff = row.exact_acc - row.acc
        auc_diff = row.exact_auc - row.auc
        ap_diff = row.exact_ap - row.ap

        acc_diff_mean_list.append(np.mean(acc_diff))
        auc_diff_mean_list.append(np.mean(auc_diff))
        ap_diff_mean_list.append(np.mean(ap_diff))

        acc_diff_std_list.append(np.std(acc_diff))
        auc_diff_std_list.append(np.std(auc_diff))
        ap_diff_std_list.append(np.std(ap_diff))

    result['acc_mean'] = np.mean(acc_mean_list)
    result['auc_mean'] = np.mean(auc_mean_list)
    result['ap_mean'] = np.mean(ap_mean_list)
    result['acc_std'] = np.mean(acc_std_list)
    result['auc_std'] = np.mean(auc_std_list)
    result['ap_std'] = np.mean(ap_std_list)

    result['acc_diff_mean'] = np.mean(acc_diff_mean_list)
    result['auc_diff_mean'] = np.mean(auc_diff_mean_list)
    result['ap_diff_mean'] = np.mean(ap_diff_mean_list)
    result['acc_diff_std'] = np.mean(acc_diff_std_list)
    result['auc_diff_std'] = np.mean(auc_diff_std_list)
    result['ap_diff_std'] = np.mean(ap_diff_std_list)

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

    acc_diff_list = []
    auc_diff_list = []
    ap_diff_list = []

    for row in gf.itertuples(index=False):

        acc_list.append(row.acc[0])
        auc_list.append(row.auc[0])
        ap_list.append(row.ap[0])

        acc_diff_list.append(row.exact_acc[0] - row.acc[0])
        auc_diff_list.append(row.exact_auc[0] - row.auc[0])
        ap_diff_list.append(row.exact_ap[0] - row.ap[0])

    result['acc_mean'] = np.mean(acc_list)
    result['auc_mean'] = np.mean(auc_list)
    result['ap_mean'] = np.mean(ap_list)
    result['acc_std'] = sem(acc_list)
    result['auc_std'] = sem(auc_list)
    result['ap_std'] = sem(ap_list)

    result['acc_diff_mean'] = np.mean(acc_diff_list)
    result['auc_diff_mean'] = np.mean(auc_diff_list)
    result['ap_diff_mean'] = np.mean(ap_diff_list)
    result['acc_diff_std'] = sem(acc_diff_list)
    result['auc_diff_std'] = sem(auc_diff_list)
    result['ap_diff_std'] = sem(ap_diff_list)

    return result


def process_retrains(gf):
    """
    Averages retrain results over multiple runs.
    """
    retrains = np.zeros(shape=(len(gf), max(args.max_depth)))

    i = 0
    for row in gf.itertuples(index=False):

        if 2 in row.retrains:
            for j in range(max(args.max_depth)):
                retrains[i][j] = row.retrains[2][j]

    retrains_mean = np.mean(retrains, axis=0)
    result = {k: v for k, v in zip(range(retrains_mean.shape[0]), retrains_mean)}
    return result


def process_results(df):
    """
    Processes utility differences, retrains, and averages the results
    over different random states.
    """
    groups = ['dataset', 'operation', 'criterion', 'n_estimators', 'max_depth',
              'max_features', 'subsample_size', 'topd', 'min_support', 'model']

    if 'epsilon' in df:
        groups += ['epsilon', 'lmbda']

    keep_cols = ['allotted_time', 'n_naive', 'n_model', 'percent_complete',
                 'n_nodes_avg', 'n_exact_avg', 'n_semi_avg']

    main_result_list = []
    retrain_result_list = []

    i = 0
    for tup, gf in tqdm(df.groupby(groups)):
        main_result = {k: v for k, v in zip(groups, tup)}
        main_result['id'] = i
        if args.periodic:
            main_result.update(process_periodic_utility(gf))
        else:
            main_result.update(process_utility(gf))
        for c in keep_cols:
            main_result[c] = gf[c].mean()
            main_result['{}_std'.format(c)] = gf[c].std()
        main_result_list.append(main_result)

        retrain_result = {'id': i}
        retrain_result.update(process_retrains(gf))
        retrain_result_list.append(retrain_result)
        i += 1

    main_df = pd.DataFrame(main_result_list)
    retrain_df = pd.DataFrame(retrain_result_list)

    return main_df, retrain_df


def create_csv(args, out_dir, logger):

    logger.info('\nGathering results...')

    experiment_settings = list(product(*[args.dataset, args.operation,
                                         args.criterion, args.rs, args.n_estimators,
                                         args.max_depth, args.max_features,
                                         args.subsample_size, args.topd, args.min_support]))

    cedar_settings = list(product(*[args.epsilon, args.lmbda]))

    results = []
    for items in tqdm(experiment_settings):
        dataset, operation, criterion, rs, n_trees, depth, max_features, sub_size, topd, support = items

        template = {'dataset': dataset, 'operation': operation, 'criterion': criterion,
                    'rs': rs, 'n_estimators': n_trees, 'max_depth': depth,
                    'max_features': max_features, 'subsample_size': sub_size, 'topd': topd,
                    'min_support': support}

        experiment_dir = os.path.join(args.in_dir, dataset, operation, criterion,
                                      'rs_{}'.format(rs),
                                      'trees_{}'.format(n_trees),
                                      'depth_{}'.format(depth),
                                      'features_{}'.format(max_features),
                                      'sub_{}'.format(sub_size),
                                      'topd_{}'.format(topd),
                                      'support_{}'.format(support))

        # skip empty experiments
        if not os.path.exists(experiment_dir):
            continue

        # add results to result dict
        dart_result = get_result(template, experiment_dir)
        if dart_result is not None:
            results.append(dart_result)

        # get cedar results
        for epsilon, lmbda in cedar_settings:
            template['epsilon'] = epsilon
            template['lmbda'] = lmbda

            extended_dir = os.path.join(experiment_dir,
                                        'epsilon_{}'.format(epsilon),
                                        'lmbda_{}'.format(lmbda))

            cedar_result = get_result(template, extended_dir)
            if cedar_result is not None:
                results.append(cedar_result)

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 180)

    df = pd.DataFrame(results)
    if 'epsilon' in df:
        df['epsilon'] = df['epsilon'].fillna(-1)
        df['lmbda'] = df['lmbda'].fillna(-1)
    logger.info('\nRaw results:\n{}'.format(df))

    logger.info('\nProcessing results...')
    main_df, retrain_df = process_results(df)
    logger.info('\nProcessed results:\n{}'.format(main_df))
    logger.info('\nRetrain results:\n{}'.format(retrain_df))

    main_fp = os.path.join(out_dir, 'results.csv')
    retrain_fp = os.path.join(out_dir, 'retrain.csv')

    main_df.to_csv(main_fp, index=None)
    retrain_df.to_csv(retrain_fp, index=None)


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
    parser.add_argument('--in_dir', type=str, default='output', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/update/csv/', help='output directory.')

    # experiment settings
    parser.add_argument('--operation', type=str, nargs='+', default=['delete', 'add'], help='update operation.')
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['surgical', 'vaccine', 'adult', 'bank_marketing', 'flight_delays', 'diabetes',
                                 'olympics', 'census', 'credit_card', 'synthetic', 'higgs'], help='dataset.')
    parser.add_argument('--criterion', type=str, nargs='+', default=['gini', 'entropy'], help='criterion.')
    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='random state.')
    parser.add_argument('--subsample_size', type=int, nargs='+', default=[1, 100, 1000], help='subsampling size.')

    # hyperparameter settings
    parser.add_argument('--n_estimators', type=int, nargs='+', default=[10, 25, 50, 100, 250, 500], help='no. trees.')
    parser.add_argument('--max_depth', type=int, nargs='+', default=[1, 3, 5, 10, 15, 20], help='max depth.')
    parser.add_argument('--max_features', type=float, nargs='+', default=[-1, 0.25], help='max features.')
    parser.add_argument('--topd', type=int, nargs='+', default=list(range(21)), help='top d.')
    parser.add_argument('--min_support', type=int, nargs='+', default=[2], help='minimum support.')
    parser.add_argument('--epsilon', type=float, nargs='+', default=[10.0, 100.0], help='epsilon.')
    parser.add_argument('--lmbda', type=float, nargs='+', default=[0.1, 0.01, 0.001, 0.0001], help='lmbda.')

    # analysis settings
    parser.add_argument('--periodic', action='store_true', default=False, help='measure periodic utility.')

    args = parser.parse_args()
    main(args)
