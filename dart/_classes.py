"""
DART (Data Addition and Removal Trees).

-The nodes in the top d layers that meet the minimum
 sample support are completely random, all other nodes
 use exact unlearning.

-Special cases:

  * [Deleting] After deleting a sample, if a decision node only
    has samples from 1 class, it turns into a leaf.

  * [Adding] After adding a sample, a non max-depth leaf node
    can turn into a decision node.
"""
import numbers

import numpy as np

from ._manager import _DataManager
from ._splitter import _Splitter
# from ._adder import _Adder
from ._remover import _Remover
from ._tree import _Tree
from ._tree import _TreeBuilder


MAX_DEPTH_LIMIT = 1000
MAX_INT = 2147483647


class Forest(object):
    """
    Random forest using Gini index as the splitting criterion.

    Parameters:
    -----------
    topd: int (default=0)
        Number of non-random layers that share the indistinguishability budget.
    k: int (default=25)
        Number of candidate thresholds per feature to consider
        through uniform sampling.
        TODO: make this auto? Maybe square root of the max. no. unique values
              across all features if this number is greater than, say, 25?
    n_estimators: int (default=100)
        Number of trees in the forest.
    max_features: int float, or str (default='sqrt')
        If int, then max_features at each split.
        If float, then max_features=int(max_features * n_features) at each split.
        If None or 'sqrt', then max_features=sqrt(n_features).
    max_depth: int (default=None)
        The maximum depth of a tree.
    criterion: str (default='gini')
        Splitting criterion to use.
    min_samples_split: int (default=2)
        The minimum number of samples needed to make a split when building a tree.
    min_samples_leaf: int (default=1)
        The minimum number of samples needed to make a leaf.
    random_state: int (default=None)
        Random state for reproducibility.
    verbose: int (default=0)
        Verbosity level.
    """
    def __init__(self,
                 topd=0,
                 k=25,
                 n_estimators=100,
                 max_features='sqrt',
                 max_depth=10,
                 criterion='gini',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 random_state=None,
                 verbose=0):
        self.topd = topd
        self.k = k
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.verbose = verbose

    def __str__(self):
        s = 'Forest:'
        s += '\ntopd={}'.format(self.topd)
        s += '\nk={}'.format(self.k)
        s += '\nn_estimators={}'.format(self.n_estimators)
        s += '\nmax_features={}'.format(self.max_features)
        s += '\nmax_depth={}'.format(self.max_depth)
        s += '\ncriterion={}'.format(self.criterion)
        s += '\nmin_samples_split={}'.format(self.min_samples_split)
        s += '\nmin_samples_leaf={}'.format(self.min_samples_leaf)
        s += '\nrandom_state={}'.format(self.random_state)
        s += '\nverbose={}'.format(self.verbose)
        return s

    def fit(self, X, y):
        """
        Build random forest.
        """
        assert X.ndim == 2
        assert y.ndim == 1
        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1]
        X, y = check_data(X, y)

        # set random state
        self.random_state_ = get_random_int(self.random_state)

        # set max_features
        if self.max_features in [-1, None, 'sqrt']:
            self.max_features_ = int(np.sqrt(self.n_features_))

        elif isinstance(self.max_features, int):
            assert self.max_features > 0
            self.max_features_ = min(self.n_features_, self.max_features)

        elif isinstance(self.max_features, float):
            assert self.max_features > 0 and self.max_features <= 1.0
            self.max_features_ = int(self.max_features * self.n_features_)

        else:
            raise ValueError('max_features {} unknown!'.format(self.max_features))

        # set max_depth
        self.max_depth_ = MAX_DEPTH_LIMIT if not self.max_depth else self.max_depth

        # set top d
        self.topd_ = min(self.topd, self.max_depth_ + 1)

        # make sure k is positive
        assert self.k > 0

        # one central location for the data
        self.manager_ = _DataManager(X, y)

        # build forest
        self.trees_ = []
        for i in range(self.n_estimators):
            # print('\n\nTree {:,}'.format(i))

            # build tree
            tree = Tree(topd=self.topd_,
                        k=self.k,
                        max_depth=self.max_depth_,
                        criterion=self.criterion,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        random_state=self.random_state_ + i,
                        verbose=self.verbose)

            tree = tree.fit(X, y, max_features=self.max_features_, manager=self.manager_)

            # add to forest
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
        X = check_data(X)

        # sum all predictions instead of storing them
        forest_preds = np.zeros(X.shape[0])
        for i, tree in enumerate(self.trees_):
            forest_preds += tree.predict_proba(X)[:, 1]

        y_mean = (forest_preds / len(self.trees_)).reshape(-1, 1)
        y_proba = np.hstack([1 - y_mean, y_mean])
        return y_proba

    # def add(self, X, y, get_indices=False):
    #     """
    #     Adds instances to the database and updates the model.
    #     """
    #     assert X.ndim == 2 and y.ndim == 1
    #     assert X.shape[1] == self.n_features_
    #     X, y = check_data(X, y)

    #     # add data to the database
    #     self.manager_.add_data(X, y)

    #     # update trees
    #     for i in range(len(self.trees_)):
    #         self.trees_[i].add()

    #     # cleanup
    #     if get_indices:
    #         return np.array(self.manager_.add_indices, dtype=np.int32)
    #     else:
    #         self.manager_.clear_add_indices()

    def delete(self, remove_indices):
        """
        Removes instances from the database and updates the model.
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

        # remove data from the database
        self.manager_.remove_data(remove_indices)

    def print(self, show_nodes=False):
        """
        Show representation of forest by showing each tree.
        """
        for tree in self.trees_:
            tree.print(show_nodes=show_nodes)

    # def clear_add_indices(self):
    #     """
    #     Delete the add index statistics.
    #     """
    #     self.manager_.clear_add_indices()

    # def get_add_retrain_depths(self):
    #     """
    #     Retrieve addition statistics.
    #     """
    #     types_list, depths_list = [], []

    #     for tree in self.trees_:
    #         types, depths = tree.get_add_retrain_depths()
    #         types_list.append(types)
    #         depths_list.append(depths)

    #     types = np.concatenate(types_list)
    #     depths = np.concatenate(depths_list)

    #     return types, depths

    def get_removal_retrain_depths(self):
        """
        Retrieve deletion statistics.
        """
        types_list, depths_list = [], []

        for tree in self.trees_:
            types, depths = tree.get_removal_retrain_depths()
            types_list.append(types)
            depths_list.append(depths)

        types = np.concatenate(types_list)
        depths = np.concatenate(depths_list)

        return types, depths

    # def get_add_retrain_sample_count(self):
    #     """
    #     Retrieve number of samples used for retrainings.
    #     """
    #     result = 0
    #     for tree in self.trees_:
    #         result += tree.get_add_retrain_sample_count()
    #     return result

    def get_removal_retrain_sample_count(self):
        """
        Retrieve number of samples used for retrainings.
        """
        result = 0
        for tree in self.trees_:
            result += tree.get_removal_retrain_sample_count()
        return result

    def get_node_statistics(self):
        """
        Return average node counts, exact node counts, and
        semi-random node counts among all trees.
        """
        counts = [tree.get_node_statistics() for tree in self.trees_]
        n_nodes, n_exact, n_semi = tuple(zip(*counts))

        n_nodes_avg = sum(n_nodes) / len(n_nodes)
        n_exact_avg = sum(n_exact) / len(n_exact)
        n_semi_avg = sum(n_semi) / len(n_semi)

        return n_nodes_avg, n_exact_avg, n_semi_avg

    def clear_removal_metrics(self):
        """
        Delete removal statistics.
        """
        for tree in self.trees_:
            tree.clear_removal_metrics()

    # def clear_add_metrics(self):
    #     """
    #     Delete add statistics.
    #     """
    #     for tree in self.trees_:
    #         tree.clear_add_metrics()

    def set_sim_mode(self, sim_mode=False):
        """
        Turns simulation mode on/off.
        """
        for tree in self.trees_:
            tree.set_sim_mode(sim_mode=sim_mode)

    def get_params(self, deep=False):
        """
        Returns the parameter of this model as a dictionary.
        """
        d = {}
        d['topd'] = self.topd
        d['k'] = self.k
        d['n_estimators'] = self.n_estimators
        d['max_features'] = self.max_features
        d['max_depth'] = self.max_depth
        d['criterion'] = self.criterion
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
    topd: int (default=0)
        Number of non-random layers that contain random nodes.
    k: int (default=25)
        No. candidate thresholds to consider through uniform sampling.
    max_depth: int (default=None)
        The maximum depth of a tree.
    criterion: str (default='gini')
        Splitting criterion to use.
    min_samples_split: int (default=2)
        The minimum number of samples needed to make a split when building a tree.
    min_samples_leaf: int (default=1)
        The minimum number of samples needed to make a leaf.
    random_state: int (default=None)
        Random state for reproducibility.
    verbose: int (default=0)
        Verbosity level.
    """
    def __init__(self,
                 topd=0,
                 k=25,
                 max_depth=10,
                 criterion='gini',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 random_state=None,
                 verbose=0):
        self.topd = topd
        self.k = k
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.verbose = verbose

    def __str__(self):
        s = 'Tree:'
        s += '\ntopd={}'.format(self.topd)
        s += '\nk={}'.format(self.k)
        s += '\nmax_depth={}'.format(self.max_depth)
        s += '\ncriterion={}'.format(self.criterion)
        s += '\nmin_samples_split={}'.format(self.min_samples_split)
        s += '\nmin_samples_leaf={}'.format(self.min_samples_leaf)
        s += '\nrandom_state={}'.format(self.random_state)
        s += '\nverbose={}'.format(self.verbose)
        return s

    def fit(self, X, y, max_features=None, manager=None):
        """
        Build decision tree.
        """
        assert X.ndim == 2
        assert y.ndim == 1

        # set random state
        self.random_state_ = check_random_state(self.random_state)

        # configure data manager
        if max_features is not None:
            assert manager is not None
            self.max_features_ = max_features
            self.manager_ = manager
            self.single_tree_ = False

        else:
            X, y = check_data(X, y)
            self.max_features_ = X.shape[1]
            self.manager_ = _DataManager(X, y)
            self.single_tree_ = True

        # set hyperparameters
        self.max_depth_ = MAX_DEPTH_LIMIT if not self.max_depth else self.max_depth
        self.topd_ = min(self.topd, self.max_depth_ + 1)
        self.use_gini_ = True if self.criterion == 'gini' else False

        # make sure k is positive
        assert self.k > 0, 'k must be greater than zero!'

        # create tree objects
        self.tree_ = _Tree()

        self.splitter_ = _Splitter(self.min_samples_leaf,
                                   self.use_gini_,
                                   self.k)

        self.tree_builder_ = _TreeBuilder(self.manager_,
                                          self.splitter_,
                                          self.min_samples_split,
                                          self.min_samples_leaf,
                                          self.max_depth_,
                                          self.topd_,
                                          self.k,
                                          self.max_features_,
                                          self.random_state_)

        self.remover_ = _Remover(self.manager_,
                                 self.tree_builder_,
                                 self.use_gini_,
                                 self.k,
                                 self.random_state_)

        # self.adder_ = _Adder(self.manager_,
        #                      self.tree_builder_,
        #                      self.use_gini_)

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
        X = check_data(X)
        y_pos = self.tree_.predict(X).reshape(-1, 1)
        y_proba = np.hstack([1 - y_pos, y_pos])
        return y_proba

    def print(self, show_nodes=False):
        """
        Shows a representation of the tree.
        """
        print('\nTREE:')
        self.tree_.print_node_count()
        self.tree_.print_node_type_count(self.lmbda, self.topd_)
        if show_nodes:
            self.tree_.print_n_samples()
            self.tree_.print_depth()
            self.tree_.print_feature()
            self.tree_.print_value()
            print()

    # def add(self, X=None, y=None, get_indices=False):
    #     """
    #     Adds instances to the database and updates the model.
    #     """

    #     if X is None:
    #         assert not self.single_tree_

    #     else:
    #         assert self.single_tree_
    #         assert X.ndim == 2 and y.ndim == 1
    #         assert X.shape[1] == self.n_features_

    #         X, y = check_data(X, y)
    #         self.manager_.add_data(X, y)

    #     # # update model
    #     # self.adder_.add(self.tree_)

    #     # cleanup
    #     if self.single_tree_:
    #         if get_indices:
    #             return np.array(self.manager_.add_indices, dtype=np.int32)
    #         else:
    #             self.manager_.clear_add_indices()

    def delete(self, remove_indices):
        """
        Removes instances from the database and updates the model.
        """

        # copy remove indices to int array
        if self.single_tree_:

            if isinstance(remove_indices, int):
                remove_indices = [remove_indices]

            if not (isinstance(remove_indices, np.ndarray) and remove_indices.dtype == np.int32):
                remove_indices = np.array(remove_indices, dtype=np.int32)

            remove_indices = np.unique(remove_indices).astype(np.int32)

        # update model
        rc = self.remover_.remove(self.tree_, remove_indices)
        if rc == -1:
            exit('Removal index invalid!')

        # remove data from the database
        if self.single_tree_:
            self.manager_.remove_data(remove_indices)

    # def clear_add_indices(self):
    #     """
    #     Delete the add index statistics.
    #     """
    #     self.manager_.clear_add_indices()

    # def get_add_retrain_depths(self):
    #     """
    #     Retrieve addition statistics.
    #     """
    #     add_types = np.array(self.adder_.add_types, dtype=np.int32)
    #     add_depths = np.array(self.adder_.add_depths, dtype=np.int32)
    #     return add_types, add_depths

    def get_removal_types_depths(self):
        """
        Retrieve deletion statistics.
        """
        remove_types = np.array(self.remover_.remove_types, dtype=np.int32)
        remove_depths = np.array(self.remover_.remove_depths, dtype=np.int32)
        return remove_types, remove_depths

    # def get_add_retrain_sample_count(self):
    #     """
    #     Return number of samples used in any retrainings.
    #     """
    #     result = self.adder_.retrain_sample_count
    #     return result

    def get_removal_retrain_sample_count(self):
        """
        Return number of samples used in any retrainings.
        """
        result = self.remover_.retrain_sample_count
        return result

    def clear_removal_metrics(self):
        """
        Delete removal statistics.
        """
        self.remover_.clear_remove_metrics()

    # def clear_add_metrics(self):
    #     """
    #     Delete add statistics.
    #     """
    #     self.adder_.clear_add_metrics()

    def get_node_statistics(self):
        """
        Retrieve:
            Total node count.
            Exact node count in the top d layers.
            Semi-random node count in the top d layers.
        """
        n_nodes = self.tree_.get_node_count()
        n_exact_nodes = self.tree_.get_exact_node_count(self.topd_)
        n_semi_random_nodes = self.tree_.get_random_node_count(self.topd_)
        return n_nodes, n_exact_nodes, n_semi_random_nodes

    def set_sim_mode(self, sim_mode=False):
        """
        Turns simulation mode on/off.
        """
        self.tree_builder_.set_sim_mode(sim_mode)

    def get_params(self, deep=False):
        """
        Returns the parameter of this model as a dictionary.
        """
        d = {}
        d['topd'] = self.topd
        d['k'] = self.k
        d['max_depth'] = self.max_depth
        d['criterion'] = self.criterion
        d['min_samples_split'] = self.min_samples_split
        d['min_samples_leaf'] = self.min_samples_leaf
        d['random_state'] = self.random_state
        d['verbose'] = self.verbose
        return d

    def set_params(self, **params):
        """
        Set the parameters of this model.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ========================================================================
# Validation Methods
# ========================================================================


def get_random_int(seed):
    """
    Get a random number from the whole range of large integer values.
    """
    np.random.seed(seed)
    return np.random.randint(MAX_INT)


# https://github.com/scikit-learn/scikit-learn/blob/\
# 95d4f0841d57e8b5f6b2a570312e9d832e69debc/sklearn/utils/validation.py#L800
def check_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance.
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand

    elif isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)

    elif isinstance(seed, np.random.RandomState):
        return seed

    else:
        raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                         ' instance' % seed)


def check_data(X, y=None):
    """
    Makes sure data is of double type,
    and labels are of integer type.
    """
    result = None

    if X.dtype != np.float32:
        X = X.astype(np.float32)

    if y is not None:
        if y.dtype != np.int32:
            y = y.astype(np.int32)
        result = X, y
    else:
        result = X

    return result
