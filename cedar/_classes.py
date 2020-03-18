"""
Decision tree implementation for binary-class classification and binary-valued attributes.
Adapted from MLFromScratch: https://github.com/eriklindernoren/ML-From-Scratch.

Uses certified removal to improve deletion efficiency.
"""
import numpy as np

from ._manager import _DataManager
from ._splitter import _Splitter
from ._remover import _Remover
from ._tree import _Tree
from ._tree import _TreeBuilder


class Forest(object):
    """
    Random forest using Gini index as the splitting criterion.

    Parameters:
    -----------
    epsilon: float (default=0.1)
        Controls the level of indistinguishability; lower for stricter gaurantees,
        higher for more deletion efficiency.
    lmbda: float (default=0.1)
        Controls the amount of noise injected into the learning algorithm.
        Set to -1 for detemrinistic trees; equivalent to setting it to infty.
    n_estimators: int (default=100)
        Number of trees in the forest.
    max_features: int float, or str (default='sqrt')
        If int, then max_features at each split.
        If float, then max_features=int(max_features * n_features) at each split.
        If 'sqrt', then max_features=sqrt(n_features).
        If None, max_features=n_features.
    max_depth: int (default=None)
        The maximum depth of a tree.
    min_samples_split: int (default=2)
        The minimum number of samples needed to make a split when building a tree.
    min_samples_leaf: int (default=1)
        The minimum number of samples needed to make a leaf.
    min_impurity_decrease: float (default=1e-8)
        The minimum impurity decrease to be considered for a split.
    random_state: int (default=None)
        Random state for reproducibility.
    verbose: int (default=0)
        Verbosity level.
    """
    def __init__(self, epsilon=0.1, lmbda=0.1, n_estimators=100, max_features='sqrt',
                 max_depth=4, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=1e-8, random_state=None, verbose=0):
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.verbose = verbose

    def __str__(self):
        s = 'Forest:'
        s += '\nepsilon={}'.format(self.epsilon)
        s += '\nlmbda={}'.format(self.lmbda)
        s += '\nn_estimators={}'.format(self.n_estimators)
        s += '\nmax_features={}'.format(self.max_features)
        s += '\nmax_depth={}'.format(self.max_depth)
        s += '\nmin_samples_split={}'.format(self.min_samples_split)
        s += '\nmin_samples_leaf={}'.format(self.min_samples_leaf)
        # s += '\nmin_impurity_decrease={}'.format(self.min_impurity_decrease)
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

        # one central location for the data
        self.manager_ = _DataManager(X, y)

        # build forest
        self.trees_ = []
        for i in range(self.n_estimators):

            if self.verbose > 2:
                print('tree {}'.format(i))

            np.random.seed(self.random_state + i)
            feature_indices = np.random.choice(self.n_features_, size=self.max_features_, replace=False)
            feature_indices = feature_indices.astype(np.int32)

            tree = Tree(epsilon=self.epsilon, lmbda=self.lmbda / self.n_estimators,
                        max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf, random_state=self.random_state + i,
                        verbose=self.verbose)
            tree = tree.fit(X, y, features=feature_indices, manager=self.manager_)
            self.trees_.append(tree)

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
        for i, tree in enumerate(self.trees_):
            forest_preds += tree.predict_proba(X)[:, 1]

        y_mean = (forest_preds / len(self.trees_)).reshape(-1, 1)
        y_proba = np.hstack([1 - y_mean, y_mean])
        return y_proba

    # TODO: adding samples can turn a leaf into a decision node
    # def add(self, X, y):
    #     """
    #     Adds instances to the training data and updates the model.
    #     """
    #     assert X.ndim == 2 and y.ndim == 1

    #     # assign index numbers to the new instances
    #     current_keys = np.fromiter(self.X_train_.keys(), dtype=np.int64)
    #     gaps = np.setdiff1d(np.arange(current_keys.max()), current_keys)
    #     if len(X) > len(gaps):
    #         extra = np.arange(current_keys.max() + 1, current_keys.max() + 1 + len(X) - len(gaps))
    #     keys = np.concatenate([gaps, extra])

    #     # add instances to the data
    #     for i, key in enumerate(keys):
    #         self.X_train_[key] = X[i]
    #         self.y_train_[key] = y[i]

    #     # add instances to each tree
    #     addition_types = []
    #     for tree in self.trees_:
    #         addition_types += tree.add(X, y, keys)

    #     return addition_types

    def delete(self, remove_indices):
        """
        Removes instances from the training data and updates the model.
        """

        # copy indices to an int array
        if isinstance(remove_indices, int):
            remove_indices = [remove_indices]

        if not (isinstance(remove_indices, np.ndarray) and remove_indices.dtype == np.int32):
            remove_indices = np.array(remove_indices, dtype=np.int32)

        remove_indices = np.unique(remove_indices).astype(np.int32)

        # update trees
        for i in range(len(self.trees_)):
            self.trees_[i].delete(remove_indices)

        # remove data from tree
        self.manager_.remove_data(remove_indices)

        # return types, depths

    def print(self, show_nodes=False, show_metadata=False):
        """
        Show representation of forest by showing each tree.
        """
        for tree in self.trees_:
            tree.print(show_nodes=show_nodes, show_metadata=show_metadata)

    def get_removal_statistics(self):
        """
        Retrieve deletion statistics.
        """
        types_list, depths_list = [], []

        for tree in self.trees_:
            types, depths = tree.get_removal_statistics()
            types_list.append(types)
            depths_list.append(depths)

        types = np.concatenate(types_list)
        depths = np.concatenate(depths_list)

        return types, depths

    def get_params(self, deep=False):
        """
        Returns the parameter of this model as a dictionary.
        """
        d = {}
        d['epsilon'] = self.epsilon
        d['lmbda'] = self.lmbda
        d['n_estimators'] = self.n_estimators
        d['max_features'] = self.max_features
        d['max_samples'] = self.max_samples
        d['max_depth'] = self.max_depth
        d['min_samples_split'] = self.min_samples_split
        d['min_samples_leaf'] = self.min_samples_leaf
        d['random_state'] = self.random_state
        d['verbose'] = self.verbose

        if deep:
            d['trees'] = {}
            for i, tree in enumerate(self.trees_):
                d['trees'][i] = tree.get_params(deep=deep)

        return d

    def set_params(self, **params):
        """
        Set the parameters of this model.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class Tree(object):
    """
    Decision Tree using Gini index for the splitting criterion.

    Parameters:
    -----------
    epsilon: float (default=0.1)
        Controls the level of indistinguishability; lower for stricter gaurantees,
        higher for more deletion efficiency.
    lmbda: float (default=0.1)
        Controls the amount of noise injected into the learning algorithm.
        Set to -1 for a detrminisic tree; equivalent to setting it to infty.
    max_depth: int (default=None)
        The maximum depth of a tree.
    min_samples_split: int (default=2)
        The minimum number of samples needed to make a split when building a tree.
    min_samples_leaf: int (default=1)
        The minimum number of samples needed to make a leaf.
    random_state: int (default=None)
        Random state for reproducibility.
    verbose: int (default=0)
        Verbosity level.
    """
    def __init__(self, epsilon=0.1, lmbda=0.1, max_depth=4, min_samples_split=2,
                 min_samples_leaf=1, random_state=None, verbose=0):
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.max_depth = -1 if max_depth is None else max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.verbose = verbose

    def __str__(self):
        s = 'Tree:'
        s += '\nepsilon={}'.format(self.epsilon)
        s += '\nlmbda={}'.format(self.lmbda)
        s += '\nmax_depth={}'.format(self.max_depth)
        s += '\nmin_samples_split={}'.format(self.min_samples_split)
        s += '\nmin_samples_leaf={}'.format(self.min_samples_leaf)
        s += '\nrandom_state={}'.format(self.random_state)
        s += '\nverbose={}'.format(self.verbose)
        return s

    def fit(self, X, y, features=None, manager=None):
        """
        Build decision tree.
        """
        assert X.ndim == 2
        assert y.ndim == 1

        if features is not None:
            assert manager is not None
            self.n_features_ = features.shape[0]
            self.manager_ = manager
            self.single_tree_ = False

        else:
            features = np.arange(X.shape[1], dtype=np.int32)
            self.n_features_ = features.shape[0]
            self.manager_ = _DataManager(X, y)
            self.single_tree_ = True

        self.tree_ = _Tree(features)
        self.splitter_ = _Splitter(self.min_samples_leaf, self.lmbda)
        self.remover_ = _Remover(self.manager_, self.epsilon, self.lmbda)
        self.tree_builder_ = _TreeBuilder(self.manager_, self.splitter_,
                                          self.min_samples_split, self.min_samples_leaf,
                                          self.max_depth, self.random_state)
        self.tree_builder_.build(self.tree_)

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
        y_pos = self.tree_.predict(X).reshape(-1, 1)
        y_proba = np.hstack([1 - y_pos, y_pos])
        return y_proba

    def print(self, show_nodes=False, show_metadata=False):
        """
        Shows a representation of the tree.
        """
        print('\nTree:')
        print('nodes: {}'.format(self.tree_.n_nodes))
        print('features: {}'.format(self.tree_.feature_indices))

        if show_nodes:
            print('counts: {}'.format(self.tree_.counts))
            print('pos counts: {}'.format(self.tree_.pos_counts))
            print('leaf values: {}'.format(self.tree_.values))
            print('p: {}'.format(self.tree_.p))
            print('feature counts: {}'.format(self.tree_.feature_counts))
            print('chosen features: {}'.format(self.tree_.chosen_features))
            print('depths: {}'.format(self.tree_.depth))
            print('left_children: {}'.format(self.tree_.left_children))
            print('right_children: {}'.format(self.tree_.right_children))

            if show_metadata:
                for i in range(self.tree_.n_nodes):
                    if self.tree_.chosen_features[i] != -2:
                        print('node {}:'.format(i))
                        print('  left counts: {}'.format(self.tree_._get_left_counts(i)))
                        print('  left pos counts: {}'.format(self.tree_._get_left_pos_counts(i)))
                        print('  right counts: {}'.format(self.tree_._get_right_counts(i)))
                        print('  right pos counts: {}'.format(self.tree_._get_right_pos_counts(i)))
                        print('  features: {}'.format(self.tree_._get_features(i)))
                    else:
                        print('node {}:'.format(i))
                        print('  leaf samples: {}'.format(self.tree_._get_leaf_samples(i)))

    # def add(self, X, y, keys=None):
    #     """
    #     Adds instances to the training data and updates the model.
    #     """
    #     assert X.ndim == 2 and y.ndim == 1

    #     # assign index numbers to the new instances
    #     if self.single_tree_:
    #         current_keys = np.fromiter(self.X_train_.keys(), dtype=np.int64)
    #         gaps = np.setdiff1d(np.arange(current_keys.max()), current_keys)
    #         if len(X) > len(gaps):
    #             extra = np.arange(current_keys.max() + 1, current_keys.max() + 1 + len(X) - len(gaps))
    #         keys = np.concatenate([gaps, extra])
    #     else:
    #         X = X[:, self.feature_indices]
    #         assert keys is not None

    #     # add instances to the data
    #     if self.single_tree_:
    #         for i, key in enumerate(keys):
    #             self.X_train_[key] = X[i]
    #             self.y_train_[key] = y[i]

    #     # update model
    #     self.addition_types_ = []
    #     self.root_ = self._add(X, y, keys)

    #     return self.addition_types_

    def delete(self, remove_indices):
        """
        Removes instance remove_ndx from the training data and updates the model.
        """

        # copy remove indices to int array
        if self.single_tree_:

            if isinstance(remove_indices, int):
                remove_indices = [remove_indices]

            if not (isinstance(remove_indices, np.ndarray) and remove_indices.dtype == np.int32):
                remove_indices = np.array(remove_indices, dtype=np.int32)

            remove_indices = np.unique(remove_indices).astype(np.int32)

        # update model
        rc = self.remover_.remove(self.tree_, self.tree_builder_, remove_indices)
        if rc == -1:
            exit('Removal index invalid!')

        # remove data
        if self.single_tree_:
            self.manager_.remove_data(remove_indices)

    def get_removal_statistics(self):
        """
        Retrieve deletion statistics.
        """
        result = self.remover_.remove_types, self.remover_.remove_depths
        return result

    def get_params(self, deep=False):
        """
        Returns the parameter of this model as a dictionary.
        """
        d = {}
        d['epsilon'] = self.epsilon
        d['lmbda'] = self.lmbda
        d['max_depth'] = self.max_depth
        d['min_samples_split'] = self.min_samples_split
        d['min_samples_leaf'] = self.min_samples_leaf
        d['random_state'] = self.random_state
        d['verbose'] = self.verbose
        d['feature_indices'] = self.feature_indices
        d['get_data'] = self.get_data

        return d

    def set_params(self, **params):
        """
        Set the parameters of this model.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # TODO: revise
    # def _add(self, X, y, add_indices, tree=None, current_depth=0, parent_p=None):

    #     # get root node of the tree
    #     if tree is None:
    #         tree = self.root_

    #     # type 1: leaf node, update its metadata
    #     if tree.value is not None:
    #         self._increment_leaf_node(tree, y, add_indices)

    #         if self.verbose > 1:
    #             print('tree check complete, ended at depth {}'.format(current_depth))

    #         self.addition_types_.append('1')
    #         return tree

    #     # decision node, update the high-level metadata
    #     tree.meta['count'] += len(y)
    #     tree.meta['pos_count'] += np.sum(y)

    #     # udpate gini_index for each attribute in this node
    #     gini_indexes = []
    #     p_ndx = None

    #     for i, attr_ndx in enumerate(tree.meta['attr']):

    #         left_indices = np.where(X[:, attr_ndx] == 1)[0]
    #         right_indices = np.setdiff1d(np.arange(X.shape[0]), left_indices)
    #         y_left, y_right = y[left_indices], y[right_indices]

    #         if len(y_left) > 0:
    #             self._increment_decision_node(tree.meta, attr_ndx, 'left', y_left)

    #         if len(y_right) > 0:
    #             self._increment_decision_node(tree.meta, attr_ndx, 'right', y_right)

    #         # recompute the gini index for this attribute
    #         attr_dict = tree.meta['attr'][attr_ndx]
    #         gini_index = self._compute_gini_index(attr_dict)
    #         gini_indexes.append(gini_index)
    #         attr_dict['gini_index'] = gini_index

    #         # get mapping from chosen attribute to distribution index
    #         if tree.feature_i == attr_ndx:
    #             p_ndx = i

    #     # get old and updated probability distributions
    #     old_p = tree.meta['p']
    #     p_dist = self._generate_distribution(gini_indexes, cur_ndx=np.argmax(old_p))
    #     p = p_dist[p_ndx] if parent_p is None else p_dist[p_ndx] * parent_p
    #     ratio = self._div1(p, old_p)

    #     # retrain if probability ratio of the chosen attribute is outside the range [e^-ep, e^ep]
    #     if ratio > np.exp(self.epsilon) or ratio < np.exp(-self.epsilon):

    #         if self.verbose > 1:
    #             print('rebuilding at depth {}'.format(current_depth))

    #         indices = self._get_indices(tree, current_depth)
    #         indices = self._add_elements(indices, add_indices)
    #         Xa, ya, keys = self.get_data(indices, self.feature_indices)
    #         self.deletion_types_.append('{}_{}'.format('2', current_depth))

    #         return self._build(Xa, ya, keys, current_depth)

    #     # continue checking the tree
    #     else:

    #         left_indices = np.where(X[:, tree.feature_i] == 1)[0]
    #         right_indices = np.setdiff1d(np.arange(X.shape[0]), left_indices)
    #         y_left, y_right = y[left_indices], y[right_indices]

    #         if len(left_indices) > 0:

    #             if self.verbose > 1:
    #                 print('check complete at depth {}, traversing left'.format(current_depth))

    #             X_left = X[left_indices]
    #             left_add_indices = add_indices[left_indices]
    #             left_branch = self._add(X_left, y_left, left_add_indices, tree=tree.left_branch,
    #                                     current_depth=current_depth + 1, parent_p=p)
    #             tree.left_branch = left_branch

    #         if len(right_indices) > 0:

    #             if self.verbose > 1:
    #                 print('check complete at depth {}, traversing right'.format(current_depth))

    #             X_right = X[right_indices]
    #             right_add_indices = add_indices[right_indices]
    #             right_branch = self._add(X_right, y_right, right_add_indices, tree=tree.right_branch,
    #                                      current_depth=current_depth + 1, parent_p=p)
    #             tree.right_branch = right_branch

    #         return tree
