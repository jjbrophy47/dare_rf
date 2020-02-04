"""
Decision tree implementation for binary classification with binary attributes.
Adapted from MLFromScratch: https://github.com/eriklindernoren/ML-From-Scratch.
"""
import numpy as np


class RF(object):
    """
    Random forest using Gini index as the splitting criterion.

    Parameters:
    -----------
    n_estimators: int (default=100)
        Number of trees in the forest.
    max_features: int float, or str (default='sqrt')
        If int, then max_features at each split.
        If float, then max_features=int(max_features * n_features) at each split.
        If 'sqrt', then max_features=sqrt(n_features).
        If None, max_features=n_features.
    max_samples: int, float (default=None)
        If None, draw n_samples samples.
        If int, draw max_samples samples.
        If float, draw int(max_samples * n_samples) samples.
    max_depth: int (default=4)
        The maximum depth of a tree.
    min_samples_split: int (default=2)
        The minimum number of samples needed to make a split when building a tree.
    random_state: int (default=None)
        Random state for reproducibility.
    verbose: int (default=0)
        Verbosity level.
    """
    def __init__(self, n_estimators=100, max_features='sqrt', max_samples=None,
                 max_depth=4, min_samples_split=2, random_state=None, verbose=0):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.verbose = verbose

    def __str__(self):
        s = 'Forest:'
        s += '\nmax_features={}'.format(self.max_features)
        s += '\nmax_samples={}'.format(self.max_samples)
        s += '\nmax_depth={}'.format(self.max_depth)
        s += '\nmin_samples_split={}'.format(self.min_samples_split)
        s += '\nrandom_state={}'.format(self.random_state)
        s += '\nverbose={}'.format(self.verbose)
        return s

    def fit(self, X, y):
        """
        Build decision tree.
        """
        assert X.ndim == 2
        assert y.ndim == 1
        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1]

        # set max_features
        if not self.max_features:
            self.max_features_ = self.n_features_

        elif self.max_features == 'sqrt':
            self.max_features_ = int(np.sqrt(self.n_features_))

        elif isinstance(self.max_features, int):
            assert self.max_features > 0
            self.max_features_ = min(self.n_features_, self.max_features)

        elif isinstance(self.max_features, float):
            assert self.max_features > 0 and self.max_features <= 1.0
            self.max_features_ = int(self.max_features * self.n_features_)

        # set max_samples
        if not self.max_samples:
            self.max_samples_ = self.n_samples_

        elif isinstance(self.max_samples, int):
            assert self.max_samples > 0
            self.max_samples_ = min(self.n_samples_, self.max_samples)

        elif isinstance(self.max_samples, float):
            assert self.max_samples > 0 and self.max_samples <= 1.0
            self.max_samples_ = int(self.max_samples * self.n_samples_)

        # build forest
        self.trees = []
        for i in range(self.n_estimators):

            if self.verbose > 0:
                print('tree {}'.format(i))

            np.random.seed(self.random_state + i)
            feature_indices = np.random.choice(self.n_features_, size=self.max_features_, replace=False)

            np.random.seed(self.random_state + i)
            sample_indices = np.random.choice(self.n_samples_, size=self.max_samples_, replace=False)

            X_sub, y_sub = X[np.ix_(sample_indices, feature_indices)], y[sample_indices]
            tree = Tree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                        verbose=self.verbose, feature_indices=feature_indices)
            tree = tree.fit(X_sub, y_sub)
            self.trees.append(tree)

        return self

    def predict(self, X):
        """
        Classify samples one by one and return the set of labels.
        """
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred

    def predict_proba(self, X):
        """
        Classify samples one by one and return the set of labels.
        """
        assert X.ndim == 2

        # sum all predictions instead of storing them
        forest_preds = np.zeros(X.shape[0])
        for i, tree in enumerate(self.trees):
            forest_preds += tree.predict_proba(X[:, tree.feature_indices])[:, 1]

        y_mean = (forest_preds / len(self.trees)).reshape(-1, 1)
        y_proba = np.hstack([1 - y_mean, y_mean])
        return y_proba


class Tree(object):
    """
    Decision Tree using Gini index for the splitting criterion.

    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    verbose: int
        Verbosity level.
    """
    def __init__(self, min_samples_split=2, max_depth=4, verbose=0, feature_indices=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.verbose = verbose
        self.feature_indices = feature_indices

    def __str__(self):
        s = 'Tree:'
        s += '\nmin_samples_split={}'.format(self.min_samples_split)
        s += '\nmax_depth={}'.format(self.max_depth)
        s += '\nverbose={}'.format(self.verbose)
        s += '\nfeature_indices={}'.format(self.feature_indices)
        return s

    def fit(self, X, y):
        """
        Build decision tree.
        """
        assert X.ndim == 2
        assert y.ndim == 1
        self.n_features_ = X.shape[1]

        self.root_ = self._build_tree(X, y)
        return self

    def _build_tree(self, X, y, current_depth=0):
        """
        Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data.
        """

        # keeps track of the best attribute
        best_gini_index = 1
        best_feature_ndx = None

        # additional data structure to maintain attribute split info
        n_samples = X.shape[0]
        pos_count = np.sum(y)

        # handle edge cases
        create_leaf = False
        if n_samples > 0:

            # all instances of the same class
            if pos_count == 0 or pos_count == n_samples:

                # the root node contains instances from the same class
                if current_depth == 0:
                    raise ValueError('root node contains only instances from the same class!')

                # create leaf
                else:
                    create_leaf = True

        else:
            raise ValueError('Zero samples in this node!, depth: {}'.format(current_depth))

        # error checking
        if n_samples >= self.min_samples_split and current_depth < self.max_depth and \
                self.n_features_ - current_depth > 0 and not create_leaf:

            # iterate through each feature
            for i in range(self.n_features_):

                # split the binary attribute
                left_indices = np.where(X[:, i] == 1)[0]
                right_indices = np.setdiff1d(np.arange(n_samples), left_indices)

                if self.verbose > 1:
                    print(i, y[left_indices], y[right_indices])

                # make sure there is atleast 1 sample in each branch
                if len(left_indices) > 0 and len(right_indices) > 0:

                    # gather stats about the split to compute the Gini index
                    left_count = len(left_indices)
                    left_pos_count = np.sum(y[left_indices])
                    right_count = n_samples - left_count
                    right_pos_count = np.sum(y[right_indices])

                    # compute the weighted Gini index of this feature
                    left_pos_prob = left_pos_count / left_count
                    left_weight = left_count / n_samples
                    left_index = 1 - (np.square(left_pos_prob) + np.square(1 - left_pos_prob))
                    left_weighted_index = left_weight * left_index

                    right_pos_prob = right_pos_count / right_count
                    right_weight = right_count / n_samples
                    right_index = 1 - (np.square(right_pos_prob) + np.square(1 - right_pos_prob))
                    right_weighted_index = right_weight * right_index

                    # compute the gini Index
                    gini_index = round(left_weighted_index + right_weighted_index, 8)

                    # keep the best attribute
                    if gini_index < best_gini_index:
                        best_gini_index = gini_index
                        best_left_indices = left_indices
                        best_right_indices = right_indices
                        best_feature_ndx = i

            # build the node with the best attribute
            if best_feature_ndx is not None:
                left_node = self._build_tree(X[best_left_indices], y[best_left_indices], current_depth + 1)
                right_node = self._build_tree(X[best_right_indices], y[best_right_indices], current_depth + 1)
                return DecisionNode(feature_i=best_feature_ndx, left_branch=left_node, right_branch=right_node)

        # we're at a leaf => determine value
        return DecisionNode(value=pos_count / n_samples)

    def copy(self):
        """
        Return a deep copy of this object.
        """
        tree = Tree(min_samples_split=self.min_samples_split, max_depth=self.max_depth, verbose=self.verbose)
        tree.n_features_ = self.n_features_
        tree.X_train_ = self.X_train_.copy()
        tree.y_train_ = self.y_train_.copy()

        # recursively copy the tree
        tree.root_ = self.root_.copy()

        return tree

    def predict(self, X):
        """
        Classify samples one by one and return the set of labels.
        """
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred

    def predict_proba(self, X):
        """
        Classify samples one by one and return the set of labels.
        """
        assert X.ndim == 2
        y_pos = self._predict_value(X).reshape(-1, 1)
        y_proba = np.hstack([1 - y_pos, y_pos])
        return y_proba

    def print_tree(self, tree=None, indent='\t', depth=0):
        """
        Recursively print the decision tree.
        """
        if tree is None:
            tree = self.root_

        indent_str = indent * (depth + 1)

        # If we're at leaf => print the label
        if tree.value is not None:
            if self.verbose > 1:
                y_vals = [self.y_train_[ndx] for ndx in tree.node_dict['indices']]
                print(tree.value, tree.node_dict['indices'], y_vals)
            else:
                print(tree.value, tree.node_dict['indices'])

        # Go deeper down the tree
        else:

            # Print test
            print("X%s? " % (tree.feature_i))

            # Print the left branch
            print("%sT->" % (indent_str), end="")
            self.print_tree(tree.left_branch, depth=depth + 1)

            # Print the right branch
            print("%sF->" % (indent_str), end="")
            self.print_tree(tree.right_branch, depth=depth + 1)

    def equals(self, other=None, this=None):
        """
        Tests if this tree is equal to another tree.
        """

        # initialize tree
        if this is None:
            this = self.root_
            if other is None:
                return 0
            else:
                other = other.root_

        # check to make sure they are both leaf nodes
        if this.value is not None:
            return 1 if this.value == other.value else 0

        # check to make sure they have the same attribute split
        if this.feature_i is not None:
            return 1 if this.feature_i == other.feature_i and \
                self.equals(this.left_branch, other.left_branch) and \
                self.equals(this.right_branch, other.right_branch) else 0

    def _predict_value(self, X, tree=None, keys=None):
        """
        Do a recursive search down the tree and make a prediction of the data sample by the
        value of the leaf that we end up at.
        """
        if tree is None:
            self.preds_ = np.empty(X.shape[0])
            keys = np.arange(X.shape[0])
            tree = self.root_

        # if we have a value (i.e we're at a leaf) => return value as the prediction
        if tree.value is not None:
            self.preds_[keys] = tree.value

        # split the binary attribute
        else:
            left_indices = np.where(X[:, tree.feature_i] == 1)[0]
            right_indices = np.setdiff1d(np.arange(X.shape[0]), left_indices)

            if len(left_indices) > 0:
                self._predict_value(X[left_indices], tree=tree.left_branch, keys=keys[left_indices])

            if len(right_indices) > 0:
                self._predict_value(X[right_indices], tree=tree.right_branch, keys=keys[right_indices])

        return self.preds_


class DecisionNode():
    """Class that represents a decision node or leaf in the decision tree

    Parameters:
    -----------
    feature_i: int
        Feature index which we want to use as the threshold measure.
    threshold: float
        The value that we will compare feature values at feature_i against to determine the prediction.
    value: float
        The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """
    def __init__(self, feature_i=None, threshold=None, value=None, left_branch=None, right_branch=None):
        self.feature_i = feature_i          # Index for the feature that is tested
        self.threshold = threshold          # Threshold value for feature
        self.value = value                  # Value if the node is a leaf in the tree
        self.left_branch = left_branch      # 'Left' subtree
        self.right_branch = right_branch    # 'Right' subtree

    def copy(self):
        left_node = self.left_branch.copy() if self.left_branch is not None else None
        right_node = self.right_branch.copy() if self.right_branch is not None else None
        node = DecisionNode(feature_i=self.feature_i, threshold=self.threshold, value=self.value,
                            left_branch=left_node, right_branch=right_node)
        return node
