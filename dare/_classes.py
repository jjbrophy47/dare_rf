"""
DaRE (Data Removal-Enabled) Trees.

-The nodes in the `topd` layers of each tree
 are completely random, all other decision nodes are greedy.

-Special cases:

  * [Deleting] After deleting a sample, if a decision node only
    has samples from 1 class, it turns into a leaf.

  * [Adding] After adding a sample, a non max-depth leaf node
    can turn into a decision node.
"""
import numbers

import numpy as np

from ._manager import _DataManager
from ._config import _Config
from ._splitter import _Splitter
from ._remover import _Remover
from ._simulator import _Simulator
from ._tree import _Tree
from ._tree import _TreeBuilder


MAX_DEPTH_LIMIT = 1000
MAX_INT = 2147483647


class Forest(object):
    """
    DaRE forest, a random forests model that can efficiently
    remove training data AFTER training.

    Parameters:
    -----------
    topd: int (default=0)
        Number of random-node layers, starting from the top.
    k: int (default=25)
        Number of candidate thresholds per feature to consider
        through uniform sampling.
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
        Build DaRE forest.
        """
        assert X.ndim == 2
        assert y.ndim == 1
        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1]
        X, y = check_data(X, y)

        # set random state
        self.random_state_ = get_random_int(self.random_state)

        # set max. features
        self.max_features_ = check_max_features(self.max_features, self.n_features_)

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

    def sim_delete(self, remove_index):
        """
        Simulate the deletion of a SINGLE example.

        Returns the number of samples that needs to be retrained
          if this example were to be deleted.
        """

        # change `remove_index` into the right data type
        if not isinstance(remove_index, np.int64):
            remove_index = np.int64(remove_index)

        # simulate a deletion for each tree
        n_samples_to_retrain = 0
        for i in range(len(self.trees_)):
            n_samples_to_retrain += self.trees_[i].sim_delete(remove_index)

        return n_samples_to_retrain

    def get_delete_metrics(self):
        """
        Retrieve deletion statistics.
        """
        types_list, depths_list, costs_list = [], [], []

        # get metrics for each tree
        for tree in self.trees_:
            types, depths, costs = tree.get_delete_metrics()
            types_list.append(types)
            depths_list.append(depths)
            costs_list.append(costs)

        # process metrics from all trees
        types = np.concatenate(types_list)
        depths = np.concatenate(depths_list)
        costs = np.concatenate(costs_list)

        return types, depths, costs

    def get_node_statistics(self):
        """
        Return average no. random, greedy, and total node counts over all trees.
        """
        n_nodes_list, n_random_nodes_list, n_greedy_nodes_list = [], [], []

        # get metrics for each tree
        for tree in self.trees_:
            n_nodes, n_random_nodes, n_greedy_nodes = tree.get_node_statistics()
            n_nodes_list.append(n_nodes)
            n_random_nodes_list.append(n_random_nodes)
            n_greedy_nodes_list.append(n_greedy_nodes)

        # take the avg. of the counts
        avg_n_nodes = np.mean(n_nodes_list)
        avg_n_random_nodes = np.mean(n_random_nodes_list)
        avg_n_greedy_nodes = np.mean(n_greedy_nodes_list)

        return avg_n_nodes, avg_n_random_nodes, avg_n_greedy_nodes

    def clear_delete_metrics(self):
        """
        Delete removal statistics.
        """
        for tree in self.trees_:
            tree.clear_delete_metrics()

    def get_memory_usage(self):
        """
        Return total memory (in bytes) used by the forest.
        """
        structure_memory = 0
        decision_stats_memory = 0
        leaf_stats_memory = 0

        # add up memory used by each tree
        for tree in self.trees_:
            struc_mem, decision_mem, leaf_mem = tree.get_memory_usage()
            structure_memory += struc_mem
            decision_stats_memory += decision_mem
            leaf_stats_memory += leaf_mem

        return structure_memory, decision_stats_memory, leaf_stats_memory

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
    Dare tree, a decision tree that can efficiently
    remove training data AFTER training.

    Parameters:
    -----------
    topd: int (default=0)
        Number of random-node layers, starting from the top.
    k: int (default=25)
        No. candidate thresholds to consider through uniform sampling.
    max_depth: int (default=None)
        The maximum depth of the tree.
    criterion: str (default='gini')
        Splitting criterion to use.
    min_samples_split: int (default=2)
        The minimum number of samples needed to make a split when building the tree.
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
        Build DaRE tree.
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

        self.config_ = _Config(self.min_samples_split,
                               self.min_samples_leaf,
                               self.max_depth_,
                               self.topd_,
                               self.k,
                               self.max_features_,
                               self.use_gini_,
                               self.random_state_)

        self.splitter_ = _Splitter(self.config_)

        self.tree_builder_ = _TreeBuilder(self.manager_,
                                          self.splitter_,
                                          self.config_)

        self.remover_ = _Remover(self.manager_,
                                 self.tree_builder_,
                                 self.config_)

        self.simulator_ = _Simulator(self.manager_,
                                     self.config_)

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

    def sim_delete(self, remove_index):
        """
        Removes instances from the database and updates the model.
        """

        # change `remove_index` into the right data type
        if self.single_tree_:
            if not isinstance(remove_index, np.int64):
                remove_index = np.int64(remove_index)

        # update model
        n_samples_to_retrain = self.simulator_.sim_delete(self.tree_, remove_index)
        if n_samples_to_retrain == -1:
            exit('Removal index invalid!')

        return n_samples_to_retrain

    def get_delete_metrics(self):
        """
        Retrieve deletion statistics.
        """
        remove_types = np.array(self.remover_.remove_types, dtype=np.int32)
        remove_depths = np.array(self.remover_.remove_depths, dtype=np.int32)
        remove_costs = np.array(self.remover_.remove_costs, dtype=np.int32)
        return remove_types, remove_depths, remove_costs

    def clear_delete_metrics(self):
        """
        Delete removal statistics.
        """
        self.remover_.clear_metrics()

    def get_node_statistics(self):
        """
        Returns the no. total nodes, no. random nodes, and no. greedy nodes.
        """
        n_nodes = self.tree_.get_node_count()
        n_random_nodes = self.tree_.get_random_node_count(self.topd_)
        n_greedy_nodes = self.tree_.get_greedy_node_count(self.topd_)
        return n_nodes, n_random_nodes, n_greedy_nodes

    def set_sim_mode(self, sim_mode=False):
        """
        Turns simulation mode on/off.
        """
        self.tree_builder_.set_sim_mode(sim_mode)

    def get_memory_usage(self):
        """
        Return total memory (in bytes) used by the tree.
        """
        structure_memory = self.tree_.get_structure_memory()
        decision_stats_memory = self.tree_.get_decision_stats_memory()
        leaf_stats_memory = self.tree_.get_leaf_stats_memory()
        return structure_memory, decision_stats_memory, leaf_stats_memory

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


def check_max_features(max_features, n_features):
    """
    Takes an int, float, or str.
      -Int > 0: Returns min(`max_features`, `n_features`)
      -Float in range (0.0, 1.0]: Returns fration of `n_features`.
      -[-1, None, 'sqrt']: Returns sqrt(`n_features`).

    Returns a valid number for the max. features, or throws an error.
    """
    assert n_features > 0

    result = None

    # return square root of no. features
    if max_features in [-1, None, 'sqrt']:
        result = int(np.sqrt(n_features))

    # may be an int, float, or string representation of an int or float
    else:
        temp_max_features = None

        # try converting to an int
        try:
            temp_max_features = int(max_features)

        # try converting to a float
        except ValueError:
            try:
                temp_max_features = float(max_features)

            except ValueError:
                pass

        if isinstance(temp_max_features, int):
            assert temp_max_features > 0
            result = min(n_features, temp_max_features)

        elif isinstance(temp_max_features, float):
            assert temp_max_features > 0 and temp_max_features <= 1.0
            result = int(temp_max_features * n_features)

        else:
            raise ValueError('max_features {} unknown!'.format(max_features))

    return result


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
