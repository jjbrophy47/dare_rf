"""
Decision tree implementation for binary-class classification and binary-valued attributes.
Adapted from MLFromScratch: https://github.com/eriklindernoren/ML-From-Scratch.

Uses certified removal to improve deletion efficiency.
"""
import numpy as np

from ._splitter import _Splitter
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
    max_samples: int, float (default=None)
        If None, draw n_samples samples.
        If int, draw max_samples samples.
        If float, draw int(max_samples * n_samples) samples.
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
                 max_samples=None, max_depth=4, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=1e-8, random_state=None, verbose=0):
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
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
        # s += '\nmax_samples={}'.format(self.max_samples)
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

        # save the data into dicts for easy deletion
        self.X_train_, self.y_train_ = self._numpy_to_dict(X, y)

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

        # # set max_samples
        # if not self.max_samples:
        #     self.max_samples_ = self.n_samples_

        # elif isinstance(self.max_samples, int):
        #     assert self.max_samples > 0
        #     self.max_samples_ = min(self.n_samples_, self.max_samples)

        # elif isinstance(self.max_samples, float):
        #     assert self.max_samples > 0 and self.max_samples <= 1.0
        #     self.max_samples_ = int(self.max_samples * self.n_samples_)

        # build forest
        self.trees_ = []
        for i in range(self.n_estimators):

            if self.verbose > 2:
                print('tree {}'.format(i))

            np.random.seed(self.random_state + i)
            feature_indices = np.random.choice(self.n_features_, size=self.max_features_, replace=False)

            # np.random.seed(self.random_state + i)
            # sample_indices = np.random.choice(self.n_samples_, size=self.n_samples, replace=False)

            X_sub, y_sub = X[:, feature_indices], y
            tree = Tree(epsilon=self.epsilon, lmbda=self.lmbda / self.n_estimators,
                        max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf, random_state=self.random_state + i,
                        feature_indices=feature_indices, verbose=self.verbose,
                        get_data=self._get_numpy_data)
            # tree = tree.fit(X_sub, y_sub, sample_indices)
            tree = tree.fit(X_sub, y_sub)
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
            forest_preds += tree.predict_proba(X[:, tree.feature_indices])[:, 1]

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

    # def delete(self, remove_indices):
    #     """
    #     Removes instances from the training data and updates the model.
    #     """
    #     if isinstance(remove_indices, int):
    #         remove_indices = np.array([remove_indices], dtype=np.int32)

    #     # delete instances from each tree
    #     deletion_types = []
    #     for tree in self.trees_:
    #         deletion_types += tree.delete(remove_indices)

    #     # remove the instances from the data
    #     for remove_ndx in remove_indices:
    #         del self.X_train_[remove_ndx]
    #         del self.y_train_[remove_ndx]

    #     return deletion_types

    def print(self, show_nodes=False, show_metadata=False):
        """
        Show representation of forest by showing each tree.
        """
        for tree in self.trees_:
            tree.print_tree(show_nodes=show_nodes, show_metadata=show_metadata)

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

    # private
    def _get_numpy_data(self, indices, feature_indices):
        """
        Collects the data from the dicts as specified by indices,
        then puts them into numpy arrays.
        """
        n_samples = len(indices)
        X = np.zeros((n_samples, self.n_features_), np.int32)
        y = np.zeros(n_samples, np.int32)
        keys = np.zeros(n_samples, np.int32)

        for i, ndx in enumerate(indices):
            X[i] = self.X_train_[ndx]
            y[i] = self.y_train_[ndx]
            keys[i] = ndx
        X = X[:, feature_indices]

        return X, y, keys

    def _numpy_to_dict(self, X, y):
        """
        Converts numpy data into dicts.
        """
        Xd, yd = {}, {}
        for i in range(X.shape[0]):
            Xd[i] = X[i]
            yd[i] = y[i]
        return Xd, yd


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
    get_data: function (parameters: indices)
        Method to retrieve data.
    feature_indices: list (default=None)
        Indices of the features present in this tree.
    """
    def __init__(self, epsilon=0.1, lmbda=0.1, max_depth=4, min_samples_split=2,
                 min_samples_leaf=1, random_state=None, verbose=0,
                 get_data=None, feature_indices=None):
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.max_depth = -1 if max_depth is None else max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.verbose = verbose
        self.feature_indices = feature_indices
        self.get_data = get_data

        self.single_tree_ = False
        if self.feature_indices is None or self.get_data is None:
            self.get_data = self._get_numpy_data
            self.single_tree_ = True

    def __str__(self):
        s = 'Tree:'
        s += '\nepsilon={}'.format(self.epsilon)
        s += '\nlmbda={}'.format(self.lmbda)
        s += '\nmax_depth={}'.format(self.max_depth)
        s += '\nmin_samples_split={}'.format(self.min_samples_split)
        s += '\nmin_samples_leaf={}'.format(self.min_samples_leaf)
        s += '\nrandom_state={}'.format(self.random_state)
        s += '\nverbose={}'.format(self.verbose)
        s += '\nfeature_indices={}'.format(self.feature_indices)
        return s

    def fit(self, X, y, keys=None):
        """
        Build decision tree.
        """
        assert X.ndim == 2
        assert y.ndim == 1
        self.n_features_ = X.shape[1]

        # make data compatible with underlying C implementation
        X = np.asfortranarray(X, dtype=np.int32)
        y = np.ascontiguousarray(y, dtype=np.int32)
        f = np.ascontiguousarray(np.arange(X.shape[1]), dtype=np.int32)

        # # save the data for easy deletion
        # if self.single_tree_:
        #     self.X_train_, self.y_train_ = self._numpy_to_dict(X, y)
        #     keys = np.arange(X.shape[0])
        # else:
        #     assert keys is not None

        self.tree_ = _Tree()
        self.splitter_ = _Splitter(self.min_samples_leaf, self.lmbda)
        self.tree_builder_ = _TreeBuilder(self.splitter_, self.min_samples_leaf,
                                          self.min_samples_split, self.max_depth,
                                          self.random_state)
        self.tree_builder_.build(self.tree_, X, y, f)

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

    def print_tree(self, show_nodes=False, show_metadata=False):
        """
        Shows a representation of the tree.
        """
        print('nodes: {}'.format(self.tree_.n_nodes))
        print('tree depth: {}'.format(self.tree_.max_depth))

        if show_nodes:
            print('leaf values: {}'.format(self.tree_.values))
            print('counts: {}'.format(self.tree_.counts))
            print('pos counts: {}'.format(self.tree_.pos_counts))
            print('feature counts: {}'.format(self.tree_.feature_counts))
            print('chosen features: {}'.format(self.tree_.chosen_features))

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

    # def delete(self, remove_indices):
    #     """
    #     Removes instance remove_ndx from the training data and updates the model.
    #     """
    #     if isinstance(remove_indices, int):
    #         remove_indices = np.array([remove_indices], dtype=np.int32)

    #     # get data to remove
    #     X, y, keys = self.get_data(remove_indices, self.feature_indices)

    #     # update model
    #     self.deletion_types_ = []
    #     self.root_ = self._delete(X, y, remove_indices)

    #     # remove the instances from the data
    #     if self.single_tree_:
    #         for remove_ndx in remove_indices:
    #             del self.X_train_[remove_ndx]
    #             del self.y_train_[remove_ndx]

    #     return self.deletion_types_

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

    # private
    # def _generate_distribution(self, gini_indexes, invalid_indices=[], cur_ndx=None):
    #     """
    #     Creates a probability distribution over the attributes given
    #     their gini index scores.
    #     """
    #     gini_indexes = np.array(gini_indexes)

    #     # numbers are too small, go into deterministic mode
    #     if np.exp(-(self.lmbda * gini_indexes.min()) / 5) == 0:
    #         p = np.where(gini_indexes == gini_indexes.min(), 1, 0)

    #         # there is a tie between 2 or more attributes
    #         if p.sum() > 1:

    #             # choose existing attribute if it is tied for first
    #             if cur_ndx and p[cur_ndx] == 1:
    #                 chosen_ndx = cur_ndx

    #             # choose the first attribute encountered that is tied for first
    #             else:
    #                 chosen_ndx = np.argmax(p == 1)

    #             # set the probability of the other attributes to zero
    #             p[np.setdiff1d(np.arange(len(p)), chosen_ndx)] = 0

    #     # create probability distribution over the attributes
    #     else:
    #         p = np.exp(-(self.lmbda * gini_indexes) / 5)
    #         if len(invalid_indices) > 0:
    #             p[np.array(invalid_indices)] = 0
    #         p = p / p.sum()

    #     return p

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

    # def _delete(self, X, y, remove_indices, tree=None, current_depth=0, parent_p=None):

    #     # get root node of the tree
    #     if tree is None:
    #         tree = self.root_

    #     # type 1a: leaf node, update its metadata
    #     if tree.value is not None:
    #         self._decrement_leaf_node(tree, y, remove_indices)

    #         if self.verbose > 1:
    #             print('tree check complete, ended at depth {}'.format(current_depth))

    #         self.deletion_types_.append('1a')
    #         return tree

    #     # decision node, update the high-level metadata
    #     count = len(y)
    #     pos_count = np.sum(y)
    #     tree.meta['count'] -= count
    #     tree.meta['pos_count'] -= pos_count

    #     # raise an error if there are only instances from one class are at the root
    #     if current_depth == 0:
    #         if tree.meta['pos_count'] == 0 or tree.meta['pos_count'] == tree.meta['count']:
    #             raise ValueError('Instances in the root node are all from the same class!')

    #     # type 1b: if remaining instances in this node are of the same class, make leaf
    #     if tree.meta['pos_count'] == 0 or tree.meta['pos_count'] == tree.meta['count']:

    #         if self.verbose > 1:
    #             print('check complete, lefotvers in the same class, creating leaf at depth {}'.format(current_depth))

    #         tree.meta['attr'] = None
    #         tree.meta['leaf_value'] = tree.meta['pos_count'] / tree.meta['count']
    #         tree.meta['indices'] = self._get_indices(tree, current_depth)
    #         tree.meta['indices'] = self._remove_elements(tree.meta['indices'], remove_indices)
    #         tree_branch = Node(value=tree.meta['leaf_value'], meta=tree.meta)
    #         self.deletion_types_.append('1b_{}'.format(current_depth))
    #         return tree_branch

    #     # type 2: all instances are removed from the left or right branch, rebuild at this node
    #     # udpate gini_index for each attribute in this node
    #     gini_indexes = []
    #     invalid_indices = []
    #     invalid_attr_indices = []
    #     n_gini_indices = 0
    #     p_ndx = None

    #     for i, attr_ndx in enumerate(tree.meta['attr']):
    #         left_indices = np.where(X[:, attr_ndx] == 1)[0]

    #         l_count = len(left_indices)
    #         r_count = count - l_count

    #         lp_count = np.sum(y[left_indices])
    #         rp_count = pos_count - lp_count

    #         # right_indices = np.setdiff1d(np.arange(X.shape[0]), left_indices)
    #         # y_left, y_right = y[left_indices], y[right_indices]

    #         gini_index = self._decrement_decision_node(tree.meta, attr_ndx, l_count, lp_count, r_count, rp_count)

    #         # if left_count > 0 or right_count > 0:
    #         #     left_status = self._decrement_decision_node(tree.node_dict, attr_ndx, 'left', y_left)

    #         # if len(y_right) > 0:
    #         #     right_status = self._decrement_decision_node(tree.node_dict, attr_ndx, 'right', y_right)

    #         # this attribute causes a hanging branch, remove it from future tree models
    #         if gini_index is None:
    #             invalid_attr_indices.append(attr_ndx)
    #             invalid_indices.append(i)
    #             gini_indexes.append(1)

    #         # recompute the gini index for this attribute
    #         else:
    #             gini_indexes.append(gini_index)
    #             tree.meta['attr'][attr_ndx]['gini_index'] = gini_index

    #             # save distribution index of chosen attribute
    #             if attr_ndx == tree.feature_i:
    #                 p_ndx = n_gini_indices

    #             n_gini_indices += 1

    #     # remove invalid attributes from the model
    #     for invalid_attr_ndx in invalid_attr_indices:
    #         del tree.meta['attr'][invalid_attr_ndx]

    #     # type 2a: the chosen feature is no longer valid
    #     if tree.feature_i in invalid_attr_indices:

    #         if self.verbose > 1:
    #             print('rebuilding at depth {}'.format(current_depth))

    #         indices = self._get_indices(tree, current_depth)
    #         indices = self._remove_elements(indices, remove_indices)
    #         Xa, ya, keys = self.get_data(indices, self.feature_indices)

    #         self.deletion_types_.append('2a_{}'.format(current_depth))
    #         return self._build(Xa, ya, keys, current_depth)

    #     # get old and updated probability distributions
    #     old_p = tree.meta['p']
    #     p_dist = self._generate_distribution(gini_indexes, invalid_indices=invalid_indices, cur_ndx=np.argmax(old_p))
    #     p = p_dist[p_ndx] if parent_p is None else p_dist[p_ndx] * parent_p
    #     ratio = self._div1(p, old_p)

    #     # type 2b: retrain if probability ratio of chosen attribute differs is outside the range [e^-ep, e^ep]
    #     if ratio < np.exp(-self.epsilon) or ratio > np.exp(self.epsilon):

    #         if self.verbose > 1:
    #             print('rebuilding at depth {}'.format(current_depth))

    #         indices = self._get_indices(tree, current_depth)
    #         indices = self._remove_elements(indices, remove_indices)
    #         Xa, ya, keys = self.get_data(indices, self.feature_indices)

    #         self.deletion_types_.append('2b_{}'.format(current_depth))
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
    #             left_remove_indices = remove_indices[left_indices]
    #             left_branch = self._delete(X_left, y_left, left_remove_indices, tree=tree.left_branch,
    #                                        current_depth=current_depth + 1, parent_p=p)
    #             tree.left_branch = left_branch

    #         if len(right_indices) > 0:

    #             if self.verbose > 1:
    #                 print('check complete at depth {}, traversing right'.format(current_depth))

    #             X_right = X[right_indices]
    #             right_remove_indices = remove_indices[right_indices]
    #             right_branch = self._delete(X_right, y_right, right_remove_indices, tree=tree.right_branch,
    #                                         current_depth=current_depth + 1, parent_p=p)
    #             tree.right_branch = right_branch

    #         return tree

    # def _increment_leaf_node(self, tree, y, add_indices):
    #     """
    #     Update this leaf node to effectively add the target indices.
    #     """
    #     meta = tree.meta
    #     meta['count'] += len(y)
    #     meta['pos_count'] += np.sum(y)
    #     meta['leaf_value'] = 0 if meta['pos_count'] == 0 else meta['pos_count'] / meta['count']
    #     meta['indices'] = self._add_elements(meta['indices'], add_indices)
    #     tree.value = meta['leaf_value']

    # def _decrement_leaf_node(self, tree, y, remove_indices):
    #     """
    #     Update this leaf node to effectively remove the target indices.
    #     """
    #     meta = tree.meta
    #     meta['count'] -= len(y)
    #     meta['pos_count'] -= np.sum(y)
    #     meta['leaf_value'] = 0 if meta['pos_count'] == 0 else meta['pos_count'] / meta['count']
    #     meta['indices'] = self._remove_elements(meta['indices'], remove_indices)
    #     tree.value = meta['leaf_value']

    # def _increment_decision_node(self, meta, attr_ndx, abranch, y):
    #     """
    #     Update the attribute dictionary of the node metadata of an addition operation.
    #     """

    #     # access the attriubute metadata
    #     abranch_dict = meta['attr'][attr_ndx][abranch]

    #     # update the affected branch
    #     abranch_dict['count'] += len(y)
    #     abranch_dict['pos_count'] += np.sum(y)
    #     abranch_dict['weight'] = abranch_dict['count'] / meta['count']
    #     abranch_dict['pos_prob'] = abranch_dict['pos_count'] / abranch_dict['count']
    #     abranch_dict['index'] = 1 - (np.square(abranch_dict['pos_prob']) + np.square(1 - abranch_dict['pos_prob']))
    #     abranch_dict['weighted_index'] = abranch_dict['weight'] * abranch_dict['index']

    #     # update the non-affected branch
    #     nabranch = 'left' if abranch == 'right' else 'right'
    #     nabranch_dict = meta['attr'][attr_ndx][nabranch]
    #     nabranch_dict['weight'] = 1 - abranch_dict['weight']
    #     nabranch_dict['weighted_index'] = nabranch_dict['weight'] * nabranch_dict['index']

    #     return True

    # def _decrement_decision_node_old(self, meta, attr_ndx, abranch, y):
    #     """
    #     Update the attribute dictionary of the node metadata from a deletion operation.
    #     """

    #     # access the attribute metadata
    #     abranch_dict = meta['attr'][attr_ndx][abranch]

    #     # only the affected instances are in this branch
    #     if abranch_dict['count'] <= len(y):
    #         return None

    #     # update the affected branch
    #     abranch_dict['count'] -= len(y)
    #     abranch_dict['pos_count'] -= np.sum(y)
    #     abranch_dict['weight'] = abranch_dict['count'] / meta['count']
    #     abranch_dict['pos_prob'] = abranch_dict['pos_count'] / abranch_dict['count']
    #     abranch_dict['index'] = 1 - (np.square(abranch_dict['pos_prob']) + np.square(1 - abranch_dict['pos_prob']))
    #     abranch_dict['weighted_index'] = abranch_dict['weight'] * abranch_dict['index']

    #     # update the non-affected branch
    #     nabranch = 'left' if abranch == 'right' else 'right'
    #     nabranch_dict = meta['attr'][attr_ndx][nabranch]
    #     nabranch_dict['weight'] = 1 - abranch_dict['weight']
    #     nabranch_dict['weighted_index'] = nabranch_dict['weight'] * nabranch_dict['index']

    #     return True

    # def _decrement_decision_node(self, meta, i, left_count, left_pos_count, right_count, right_pos_count):
    #     """
    #     Update the attribute dictionary of the node metadata from a deletion operation.
    #     """

    #     # access metadata
    #     left_meta = meta['attr'][i]['left']
    #     right_meta = meta['attr'][i]['right']

    #     # deleting causes a hanging branch
    #     if left_meta['count'] <= left_count or right_meta['count'] <= right_count:
    #         return None

    #     # update metadata
    #     left_meta['count'] -= left_count
    #     left_meta['pos_count'] -= left_pos_count
    #     right_meta['count'] -= right_count
    #     right_meta['pos_count'] -= right_pos_count

        # # recompute gini index
        # return self._gini_index(left_meta['count'], left_meta['pos_count'],
        #                         right_meta['count'], right_meta['pos_count'])

        # # access the attribute metadata
        # abranch_dict = node_dict['attr'][attr_ndx][abranch]

        # # only the affected instances are in this branch
        # if abranch_dict['count'] <= len(y):
        #     return None

        # # update the affected branch
        # abranch_dict['count'] -= len(y)
        # abranch_dict['pos_count'] -= np.sum(y)
        # abranch_dict['weight'] = abranch_dict['count'] / node_dict['count']
        # abranch_dict['pos_prob'] = abranch_dict['pos_count'] / abranch_dict['count']
        # abranch_dict['index'] = 1 - (np.square(abranch_dict['pos_prob']) + np.square(1 - abranch_dict['pos_prob']))
        # abranch_dict['weighted_index'] = abranch_dict['weight'] * abranch_dict['index']

        # # update the non-affected branch
        # nabranch = 'left' if abranch == 'right' else 'right'
        # nabranch_dict = node_dict['attr'][attr_ndx][nabranch]
        # nabranch_dict['weight'] = 1 - abranch_dict['weight']
        # nabranch_dict['weighted_index'] = nabranch_dict['weight'] * nabranch_dict['index']

        # return True

    # def _gini_index(self, n_samples, left_count, left_pos_count, right_count, right_pos_count):
    #     """
    #     Compute the gini index given the necessary information from both branches.
    #     """
    #     left_pos_prob = left_pos_count / left_count
    #     left_weight = left_count / n_samples
    #     left_index = 1 - np.square(left_pos_prob) - np.square(1 - left_pos_prob)
    #     left_weighted_index = left_weight * left_index

    #     right_pos_prob = right_pos_count / right_count
    #     right_weight = right_count / n_samples
    #     right_index = 1 - np.square(right_pos_prob) - np.square(1 - right_pos_prob)
    #     right_weighted_index = right_weight * right_index

    #     gini_index = left_weighted_index + right_weighted_index
    #     return round(gini_index, 8)

    # def _gini_index(self, attr_dict):
    #     """
    #     Compute the gini index given the weighted indexes from both branches.
    #     """
    #     gini_index = attr_dict['left']['weighted_index'] + attr_dict['right']['weighted_index']
    #     return round(gini_index, 8)

    # def _get_indices(self, tree=None, depth=0):
    #     """
    #     Recursively retrieve all the indices for this node from the leaves.
    #     """
    #     if tree is None:
    #         tree = self.root_

    #     # made it to a leaf node, return the indices
    #     if tree.value is not None:
    #         return tree.meta['indices']

    #     else:
    #         left_indices = self._get_indices(tree.left_branch, depth + 1)
    #         right_indices = self._get_indices(tree.right_branch, depth + 1)
    #         return np.concatenate([left_indices, right_indices])

    # def _predict_value(self, X, tree=None, keys=None):
    #     """
    #     Do a recursive search down the tree and make a prediction of the data sample by the
    #     value of the leaf that we end up at.
    #     """
    #     if tree is None:
    #         self.preds_ = np.empty(X.shape[0])
    #         keys = np.arange(X.shape[0])
    #         tree = self.root_

    #     # if we have a value (i.e we're at a leaf) => return value as the prediction
    #     if tree.value is not None:
    #         self.preds_[keys] = tree.value

    #     # split the binary attribute
    #     else:
    #         left_indices = np.where(X[:, tree.feature_i] == 1)[0]
    #         right_indices = np.setdiff1d(np.arange(X.shape[0]), left_indices)

    #         if len(left_indices) > 0:
    #             self._predict_value(X[left_indices], tree=tree.left_branch, keys=keys[left_indices])

    #         if len(right_indices) > 0:
    #             self._predict_value(X[right_indices], tree=tree.right_branch, keys=keys[right_indices])

    #     return self.preds_

    # def _add_elements(self, arr, elements):
    #     """
    #     Add elements to array.
    #     """
    #     return np.concatenate([arr, elements])

    # def _remove_elements(self, arr, elements):
    #     """
    #     Remove elements from array.
    #     """
    #     return np.setdiff1d(arr, elements)

    def _get_numpy_data(self, indices, feature_indices=None):
        """
        Collects the data from the dicts as specified by indices,
        then puts them into numpy arrays.
        """
        n_samples = len(indices)
        X = np.zeros((n_samples, self.n_features_), np.int32)
        y = np.zeros(n_samples, np.int32)
        keys = np.zeros(n_samples, np.int32)

        for i, ndx in enumerate(indices):
            X[i] = self.X_train_[ndx]
            y[i] = self.y_train_[ndx]
            keys[i] = ndx

        return X, y, keys

    def _numpy_to_dict(self, X, y):
        """
        Converts numpy data into dicts.
        """
        Xd, yd = {}, {}
        for i in range(X.shape[0]):
            Xd[i] = X[i]
            yd[i] = y[i]
        return Xd, yd

    # def _div1(self, a, b):
    #     """
    #     Returns 1 if the denominator is zero.
    #     """
    #     return np.divide(a, b, out=np.ones_like(a, dtype=np.float64), where=b != 0)


# class Node():
#     """
#     Class that represents a decision node or leaf in the decision tree

#     Parameters:
#     -----------
#     feature: int
#         Feature index which we want to use as the threshold measure.
#     value: float
#         The class prediction if classification tree, or float value if regression tree.
#     left: Node
#         Next node for samples where features value met the threshold.
#     right: Node
#         Next node for samples where features value did not meet the threshold.
#     meta: dict
#         Dictionary to store metatadata for easy updating.
#     """
#     def __init__(self, feature=None, value=None, left=None, right=None, meta=None):
#         self.feature = feature
#         self.value = value
#         self.left = left
#         self.right = right
#         self.meta = meta

#     def __str__(self):
#         s = 'feature={}'.format(self.feature)
#         s += '\nvalue={}'.format(self.value)
#         s += '\nleft={}'.format(self.left)
#         s += '\nright={}'.format(self.right)
#         s += '\nmeta={}'.format(self.meta)
#         return s
