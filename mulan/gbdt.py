"""
Binary GBDT implementation using CART regression trees. MSE is used to find the best split attribute and value
as this is equivalent to finding the maximum variance reduction.
Adapted from MLFromScratch: https://github.com/eriklindernoren/ML-From-Scratch
"""
import numpy as np

from gbdt_tree import Tree


class GradientBoosting:

    """
    Super class of GradientBoostingClassifier and GradientBoostinRegressor. 
    Uses a collection of regression trees that trains on predicting the gradient
    of the loss function. 

    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    learning_rate: float
        The step length that will be taken when following the negative gradient during training.
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    max_depth: int
        The maximum depth of a tree.
    """

    def __init__(self, n_estimators=100, learning_rate=0.01, min_samples_split=2, max_depth=4):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

        # Initialize regression trees
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(Tree(min_samples_split=self.min_samples_split, max_depth=self.max_depth))

    def __str__(self):
        s = 'GBDT:'
        s += '\nn_estimators={}'.format(self.n_estimators)
        s += '\nlearning_rate={}'.format(self.learning_rate)
        s += '\nmin_samples_split={}'.format(self.min_samples_split)
        s += '\nmax_depth={}'.format(self.max_depth)
        return s

    def fit(self, X, y):
        assert X.ndim == 2
        assert y.ndim == 1
        assert len(np.unique(y)) == 2
        assert X.shape[0] == len(y)

        y_pred = np.full(np.shape(y), np.mean(y))
        for i in range(self.n_estimators):
            gradient = self._gradient(y, y_pred)
            self.trees[i].fit(X, gradient)

            # Update y prediction
            update = self.trees[i].predict(X)
            y_pred -= np.multiply(self.learning_rate, update)

        return self

    def predict(self, X):
        assert X.ndim == 2

        y_pred = np.array([])

        # Make predictions
        for tree in self.trees:
            update = tree.predict(X)
            update = np.multiply(self.learning_rate, update)
            y_pred = -update if not y_pred.any() else y_pred - update

        y_pred = self._sigmoid(y_pred)
        y_pred = y_pred.reshape(-1, 1)
        y_pred = np.hstack([1 - y_pred, y_pred])

        return y_pred

    def equals(self, other):
        result = 0

        if self.n_estimators != other.n_estimators:
            return False

        # check trees
        for i in range(self.n_estimators):
            result += self.trees[i].equals(other=other.trees[i])

        # checksum
        if result != self.n_estimators:
            return False

        return True

    def print_trees(self):
        for i, tree in enumerate(self.trees):
            print('\nTree {}'.format(i + 1))
            tree.print_tree()

    def _gradient(self, y, y_pred):
        return -(y - y_pred)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    X = np.random.randn(100, 2)
    y = np.random.randint(2, size=100)
    print(X[:5], y[:5])

    g = GradientBoosting(n_estimators=5, max_depth=4, learning_rate=0.5).fit(X, y)
    g2 = GradientBoosting(n_estimators=5, max_depth=4, learning_rate=0.5).fit(X, y)
    g.print_trees()
