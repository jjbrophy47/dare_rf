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


# def process_periodic_utility(gf):
#     """
#     Processes utility differences BEFORE and DURING addition/deletion,
#     and averages the results over different random states.
#     """
#     result = {}

#     acc_mean_list, acc_std_list = [], []
#     auc_mean_list, auc_std_list = [], []
#     ap_mean_list, ap_std_list = [], []

#     acc_diff_mean_list, acc_diff_std_list = [], []
#     auc_diff_mean_list, auc_diff_std_list = [], []
#     ap_diff_mean_list, ap_diff_std_list = [], []

#     deletion_time_list = []

#     for row in gf.itertuples(index=False):

#         acc_mean_list.append(np.mean(row.acc))
#         auc_mean_list.append(np.mean(row.auc))
#         ap_mean_list.append(np.mean(row.ap))

#         acc_std_list.append(np.std(row.acc))
#         auc_std_list.append(np.std(row.auc))
#         ap_std_list.append(np.std(row.ap))

#         acc_diff = row.exact_acc - row.acc
#         auc_diff = row.exact_auc - row.auc
#         ap_diff = row.exact_ap - row.ap

#         acc_diff_mean_list.append(np.mean(acc_diff))
#         auc_diff_mean_list.append(np.mean(auc_diff))
#         ap_diff_mean_list.append(np.mean(ap_diff))

#         acc_diff_std_list.append(np.std(acc_diff))
#         auc_diff_std_list.append(np.std(auc_diff))
#         ap_diff_std_list.append(np.std(ap_diff))

#         deletion_time_list.append(row.allotted_time / row.n_model)

#     result['acc_mean'] = np.mean(acc_mean_list)
#     result['auc_mean'] = np.mean(auc_mean_list)
#     result['ap_mean'] = np.mean(ap_mean_list)
#     result['acc_std'] = np.mean(acc_std_list)
#     result['auc_std'] = np.mean(auc_std_list)
#     result['ap_std'] = np.mean(ap_std_list)
#     result['deletion_time_mean'] = np.mean(deletion_time_list)

#     result['acc_diff_mean'] = np.mean(acc_diff_mean_list)
#     result['auc_diff_mean'] = np.mean(auc_diff_mean_list)
#     result['ap_diff_mean'] = np.mean(ap_diff_mean_list)
#     result['acc_diff_std'] = np.mean(acc_diff_std_list)
#     result['auc_diff_std'] = np.mean(auc_diff_std_list)
#     result['ap_diff_std'] = np.mean(ap_diff_std_list)
#     result['deletion_time_std'] = sem(deletion_time_list)

    return result


def process_utility(gf):
    """
    Processes utility differences BEFORE deletion,
    and averages the results over different random states.
    """
    result = {}

    model_acc_list = []
    model_auc_list = []
    model_ap_list = []

    acc_diff_list = []
    auc_diff_list = []
    ap_diff_list = []

    model_delete_time_list = []

    for row in gf.itertuples(index=False):

        # extract model predictive performance
        model_acc_list.append(row.model_acc)
        model_auc_list.append(row.model_auc)
        model_ap_list.append(row.model_ap)

        # compare model predictive performance to naive
        acc_diff_list.append(row.naive_acc - row.model_acc)
        auc_diff_list.append(row.naive_auc - row.model_auc)
        ap_diff_list.append(row.naive_ap - row.model_ap)

        # record avg. deletion time for the model
        model_delete_time_list.append(row.naive_avg_delete_time / row.model_n_deleted)

    # compute mean and sem for predictive performances
    result['model_acc_mean'] = np.mean(acc_list)
    result['model_auc_mean'] = np.mean(auc_list)
    result['model_ap_mean'] = np.mean(ap_list)
    result['model_acc_sem'] = sem(acc_list)
    result['model_auc_sem'] = sem(auc_list)
    result['model_ap_sem'] = sem(ap_list)
    result['model_delete_time_mean'] = np.mean(deletion_time_list)
    result['model_delete_time_sem'] = sem(deletion_time_list)

    # compute mean and sem for predictive performance differences
    result['acc_diff_mean'] = np.mean(acc_diff_list)
    result['auc_diff_mean'] = np.mean(auc_diff_list)
    result['ap_diff_mean'] = np.mean(ap_diff_list)
    result['acc_diff_sem'] = sem(acc_diff_list)
    result['auc_diff_sem'] = sem(auc_diff_list)
    result['ap_diff_sem'] = sem(ap_diff_list)

    return result


def process_retrains(gf):
    """
    Averages retrain results over multiple runs.
    """
    retrains = np.zeros(shape=(len(gf), max(args.max_depth)))

    i = 0
    for row in gf.itertuples(index=False):

        if 1 in row.retrains:
            for j in range(max(args.max_depth)):
                retrains[i][j] = row.retrains[1][j]

    retrains_mean = np.mean(retrains, axis=0)
    result = {k: v for k, v in zip(range(retrains_mean.shape[0]), retrains_mean)}

    return result


def process_results(df):
    """
    Processes utility differences, retrains, and averages the results
    over different random states.
    """
    setting_cols = ['dataset', 'criterion', 'n_estimators', 'max_depth',
                    'topd', 'k', 'subsample_size']

    keep_cols = ['naive_avg_delete_time',
                 'naive_n_deleted',
                 'model_n_deleted',
                 'model_train_%_deleted']
                 # 'n_nodes_avg',
                 # 'n_exact_avg',
                 # 'n_semi_avg']

    # result containers
    main_result_list = []
    # retrain_result_list = []

    # loop through each experiment setting
    i = 0
    for tup, gf in tqdm(df.groupby(setting_cols)):

        # create main result
        main_result = {k: v for k, v in zip(setting_cols, tup)}
        main_result['id'] = i
        main_result.update(process_utility(gf))
        for c in keep_cols:
            main_result[c] = gf[c].mean()
            main_result['{}_std'.format(c)] = gf[c].std()
        main_result_list.append(main_result)

        # # create retrain result
        # retrain_result = {'id': i}
        # retrain_result.update(process_retrains(gf))
        # retrain_result_list.append(retrain_result)
        i += 1

    # compile results
    main_df = pd.DataFrame(main_result_list)
    # retrain_df = pd.DataFrame(retrain_result_list)

    # return main_df, retrain_df

    return main_df



def create_csv(args, out_dir, logger):

    logger.info('\nGathering results...')

    experiment_settings = list(product(*[args.dataset, args.criterion, args.rs, args.n_estimators,
                                         args.max_depth, args.topd, args.k, args.subsample_size]))

    # cedar_settings = list(product(*[args.epsilon, args.lmbda]))

    results = []
    for items in tqdm(experiment_settings):
        dataset, criterion, rs, n_trees, depth, topd, k, sub_size = items

        template = {'dataset': dataset,
                    'criterion': criterion,
                    'rs': rs,
                    'n_estimators': n_trees,
                    'max_depth': depth,
                    'topd': topd,
                    'k': k,
                    'subsample_size': sub_size}

        experiment_dir = os.path.join(args.in_dir, dataset, criterion,
                                      'rs_{}'.format(rs),
                                      'trees_{}'.format(n_trees),
                                      'depth_{}'.format(depth),
                                      'topd_{}'.format(topd),
                                      'k_{}'.format(k),
                                      'sub_{}'.format(sub_size))

        # skip empty experiments
        if not os.path.exists(experiment_dir):
            continue

        # add results to result dict
        dart_result = get_result(template, experiment_dir)
        if dart_result is not None:
            results.append(dart_result)

        # get cedar results
        # for epsilon, lmbda in cedar_settings:
        #     template['epsilon'] = epsilon
        #     template['lmbda'] = lmbda

        #     extended_dir = os.path.join(experiment_dir,
        #                                 'epsilon_{}'.format(epsilon),
        #                                 'lmbda_{}'.format(lmbda))

        #     cedar_result = get_result(template, extended_dir)
        #     if cedar_result is not None:
        #         results.append(cedar_result)

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 180)

    df = pd.DataFrame(results)
    # if 'epsilon' in df:
    #     df['epsilon'] = df['epsilon'].fillna(-1)
    #     df['lmbda'] = df['lmbda'].fillna(-1)
    logger.info('\nRaw results:\n{}'.format(df))

    logger.info('\nProcessing results...')
    main_df = process_results(df)
    logger.info('\nProcessed results:\n{}'.format(main_df))
    # logger.info('\nRetrain results:\n{}'.format(retrain_df))

    main_fp = os.path.join(out_dir, 'results.csv')
    # retrain_fp = os.path.join(out_dir, 'retrain.csv')

    main_df.to_csv(main_fp, index=None)
    # retrain_df.to_csv(retrain_fp, index=None)


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
    parser.add_argument('--out_dir', type=str, default='output/delete/csv/', help='output directory.')

    # experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['surgical', 'vaccine', 'adult', 'bank_marketing', 'flight_delays', 'diabetes',
                                 'olympics', 'census', 'credit_card', 'no_show', 'twitter' 'synthetic',
                                 'higgs', 'ctr'], help='dataset.')
    parser.add_argument('--criterion', type=str, nargs='+', default=['gini', 'entropy'], help='criterion.')
    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='random state.')
    parser.add_argument('--subsample_size', type=int, nargs='+', default=[1, 1000], help='subsampling size.')

    # hyperparameter settings
    parser.add_argument('--n_estimators', type=int, nargs='+', default=[10, 50, 100, 250], help='no. trees.')
    parser.add_argument('--max_depth', type=int, nargs='+', default=[1, 3, 5, 10, 20], help='max depth.')
    parser.add_argument('--topd', type=int, nargs='+', default=list(range(21)), help='top d.')
    parser.add_argument('--k', type=int, nargs='+', default=[5, 10, 25, 50, 100], help='no. thresholds.')

    # analysis settings
    parser.add_argument('--periodic', action='store_true', default=False, help='measure periodic utility.')

    args = parser.parse_args()
    main(args)
