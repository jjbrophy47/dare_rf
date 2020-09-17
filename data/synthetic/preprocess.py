"""
Generates a binary attribute binary classification dataset.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import KBinsDiscretizer


def main(random_state=1,
         test_size=0.2,
         n_samples=1000000,
         n_features=40,
         n_informative=5,
         n_redundant=5,
         n_repeated=0,
         n_clusters_per_class=2,
         flip_y=0.05):

    # retrieve dataset
    data = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_informative,
                               n_redundant=n_redundant,
                               n_repeated=n_repeated,
                               n_clusters_per_class=n_clusters_per_class,
                               flip_y=flip_y,
                               random_state=random_state)
    X, y = data
    indices = np.arange(len(X))

    np.random.seed(random_state)
    train_indices = np.random.choice(indices, size=int(len(X) * (1 - test_size)), replace=False)
    test_indices = np.setdiff1d(indices, train_indices)

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    kbd = KBinsDiscretizer(n_bins=5, encode='onehot-dense')
    X_train = kbd.fit_transform(X_train)
    X_test = kbd.transform(X_test)

    # cleanup
    train = np.hstack([X_train, y_train.reshape(-1, 1)])
    test = np.hstack([X_test, y_test.reshape(-1, 1)])
    train = train.astype(np.int32)
    test = test.astype(np.int32)

    print(train.shape, train[:, -1].sum())
    print(test.shape, test[:, -1].sum())

    if train[:, -1].sum() == 0 or train[:, -1].sum() == len(train):
        print('train only contains 1 class!')
    if test[:, -1].sum() == 0 or test[:, -1].sum() == len(test):
        print('test only contains 1 class!')

    # save to numpy format
    print('saving to train.npy...')
    np.save('train.npy', train)
    print('saving to test.npy...')
    np.save('test.npy', test)


if __name__ == '__main__':
    main()
