"""
Generates binary attribute data correlated with
a binary classification label.
"""
import os
import sys
import argparse

import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../..')
from utility import print_util


def _abbr(n):
    """
    Shortens numbers by putting an abbreviation.
    Only works for numbers with a digit and then a bunch of zeros.
    n : int
    ex: 1000000 -> 1m; 10000 -> 10k
    """
    suffixes = {0: '', 1: 'k', 2: 'm', 3: 'b', 4: 't'}
    s = str(n)
    prefix = s[0]
    n_zeros = len(s) - 1
    level = int(n_zeros / 3)
    extra_zeros = n_zeros % 3
    new_s = '{}{}{}'.format(prefix, '0' * extra_zeros, suffixes[level])
    return new_s


def main(args):

    # give dataset a name
    if args.name is None:
        name = 'n{}_a{}'.format(_abbr(args.n_samples), _abbr(args.n_attributes))
    else:
        name = args.name

    # create new dataset directory
    dataset_dir = os.path.join(args.out_dir, name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    logger = print_util.get_logger(os.path.join(dataset_dir, 'log.txt'))

    logger.info(args)
    logger.info('# instances: {}'.format(_abbr(args.n_samples)))
    logger.info('# attributes: {}'.format(_abbr(args.n_attributes)))

    # generate label data
    np.random.seed(args.rs)
    y = np.random.binomial(n=1, p=args.p_pos, size=args.n_samples)
    y_inv = 1 - y
    logger.info('positive label: {:.2f}'.format(y.sum() / len(y)))

    # generate attribute data
    data = []
    for i in range(args.n_attributes):

        # choose values from y a fraction of the time
        np.random.seed(args.rs + i)
        corr_frac = np.random.uniform(args.min_corr, args.max_corr)
        corr_sample = int(len(y) * corr_frac)
        logger.info('x{}-y correlation: {:.2f}'.format(i, corr_frac))

        # sample from label array
        np.random.seed(args.rs + i)
        idx = np.random.choice(np.arange(len(y)), size=corr_sample, replace=False)
        idx_inv = np.setdiff1d(np.arange(len(y)), idx)

        # fill in the data
        x = np.zeros(len(y))
        x[idx] = y[idx]
        x[idx_inv] = y_inv[idx_inv]

        data.append(x.reshape(-1, 1))
    data.append(y.reshape(-1, 1))

    # split data into train and test
    data = np.hstack(data)
    split_ndx = len(data) - int(len(data) * args.test_frac)
    train = data[:split_ndx].astype(np.int64)
    test = data[split_ndx:].astype(np.int64)
    logger.info('split into train ({}) and test ({})'.format(_abbr(len(train)), _abbr(len(test))))

    # save the numpy data
    logger.info('saving data to {}...'.format(dataset_dir))
    np.save(os.path.join(dataset_dir, 'train.npy'), train)
    np.save(os.path.join(dataset_dir, 'test.npy'), test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='data/synthetic/', help='output directory.')
    parser.add_argument('--name', default=None, help='dataset to use for the experiment.')
    parser.add_argument('--n_samples', type=int, default=1000, help='number of samples to generate.')
    parser.add_argument('--n_attributes', type=int, default=10, help='number of attributes to generate.')
    parser.add_argument('--p_pos', type=float, default=0.5, help='probability of positive label.')
    parser.add_argument('--min_corr', type=float, default=0.4, help='minimum correlation between x and y.')
    parser.add_argument('--max_corr', type=float, default=0.6, help='maximum correlation between x and y.')
    parser.add_argument('--test_frac', type=float, default=0.3, help='fraction of data to use for test.')
    parser.add_argument('--rs', type=int, default=1, help='seed to enhance reproducibility.')
    args = parser.parse_args()
    main(args)
