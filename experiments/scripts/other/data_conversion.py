"""
Test to see how long it takes to convert numpy arrays into dicts.

Array to dicts:
Example: 1M samples w/ 500 attributes: 0.610s
Example: 1M samples w/ 1k attributes: 0.739s.
Example: 1M samples w/ 2k attributes: 0.761s.
Example: 1M samples w/ 5k attributes: 0.836s.
Example: 2M samples w/ 500 attributes: 1.520s.
Example: 4M samples w/ 500 attributes: 3.222s.
Example: 10M samples w/ 500 attributes: 7.847s.

Dicts to Array:
Example: 1M samples w/ 500 attributes: 0.826s.
Example: 1M samples w/ 1k attributes: 0.971s.
Example: 1M samples w/ 2k attributes: 1.392s.
Example: 1M samples w/ 5k attributes: 2.842s.
Example: 2M samples w/ 500 attributes: 1.748s.
Example: 4M samples w/ 500 attributes: 3.288s.
Example: 10M samples w/ 500 attributes: 10.265s.
"""
import time
import argparse

import numpy as np


def _numpy_to_dict(X, y):
    Xd, yd = {}, {}
    for i in range(X.shape[0]):
        Xd[i] = X[i]
        yd[i] = y[i]
    return Xd, yd


def _dict_to_numpy(Xd, yd):
    n_attributes = len(Xd[next(iter(Xd))])
    n_samples = len(Xd)

    X = np.zeros((n_samples, n_attributes), np.int32)
    y = np.zeros(n_samples, np.int32)
    keys = np.zeros(n_samples, np.int32)

    i = 0
    for k in Xd.keys():
        X[i] = Xd[i]
        y[i] = yd[i]
        keys[i] = k

    return X, y, keys


def main(args):

    # create data
    start = time.time()
    np.random.seed(args.seed)
    X = np.random.randint(2, size=(args.n_samples, args.n_attributes))
    np.random.seed(args.seed)
    y = np.random.randint(2, size=args.n_samples)
    print('creation time: {:.3f}s'.format(time.time() - start))

    # convert data to dicts
    start = time.time()
    Xd, yd = _numpy_to_dict(X, y)
    print('numpy to dict time: {:.3f}s'.format(time.time() - start))

    # convert data to numpy
    start = time.time()
    _dict_to_numpy(Xd, yd)
    print('dicts to numpy time: {:.3f}s'.format(time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=10, help='number of samples to generate.')
    parser.add_argument('--n_attributes', type=int, default=4, help='number of attributes to generate.')
    parser.add_argument('--seed', type=int, default=423, help='seed to populate the data.')
    args = parser.parse_args()
    print(args)
    main(args)
