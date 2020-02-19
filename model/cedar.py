"""
Decision tree implementation for binary-class classification and binary-valued attributes.
Adapted from MLFromScratch: https://github.com/eriklindernoren/ML-From-Scratch.

Uses certified removal to improve deletion efficiency.
"""
import numpy as np


class RF(object):
    """
    Random forest using Gini index as the splitting criterion.

    Parameters:
    -----------
    epsilon: float (default=0.1)
        Controls the level of indistinguishability; lower for stricter gaurantees,
        higher for more deletion efficiency.
    lmbda: float (default=0.1)
        Controls the amount of noise injected into the learning algorithm.
    gamma: float (default=0.1)
        Fraction of data to support for removal.
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
    def __init__(self, epsilon=0.1, lmbda=0.1, gamma=0.1, n_estimators=100, max_features='sqrt', max_samples=None,
                 max_depth=4, min_samples_split=2, random_state=None, verbose=0):
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.verbose = verbose

    def __str__(self):
        s = 'Forest:'
        s += '\nepsilon={}'.format(self.epsilon)
        s += '\nlmbda={}'.format(self.lmbda)
        s += '\ngamma={}'.format(self.gamma)
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
            tree = Tree(epsilon=self.epsilon, lmbda=self.lmbda, gamma=self.gamma, max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split, random_state=self.random_state,
                        feature_indices=feature_indices, verbose=self.verbose, get_data=self._get_numpy_data)
            tree = tree.fit(X_sub, y_sub, sample_indices)
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

    def add(self, X, y):
        """
        Adds instances to the training data and updates the model.
        """
        assert X.ndim == 2 and y.ndim == 1

        # assign index numbers to the new instances
        current_keys = np.fromiter(self.X_train_.keys(), dtype=np.int64)
        gaps = np.setdiff1d(np.arange(current_keys.max()), current_keys)
        if len(X) > len(gaps):
            extra = np.arange(current_keys.max() + 1, current_keys.max() + 1 + len(X) - len(gaps))
        keys = np.concatenate([gaps, extra])

        # add instances to the data
        for i, key in enumerate(keys):
            self.X_train_[key] = X[i]
            self.y_train_[key] = y[i]

        # add instances to each tree
        addition_types = []
        for tree in self.trees:
            addition_types += tree.add(X, y, keys)

        return addition_types

    def delete(self, remove_indices):
        """
        Removes instances from the training data and updates the model.
        """
        if isinstance(remove_indices, int):
            remove_indices = np.array([remove_indices], dtype=np.int32)

        # delete instances from each tree
        deletion_types = []
        for tree in self.trees:
            deletion_types += tree.delete(remove_indices)

        # remove the instances from the data
        for remove_ndx in remove_indices:
            del self.X_train_[remove_ndx]
            del self.y_train_[remove_ndx]

        return deletion_types

    # private
    def _get_numpy_data(self, indices):
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


class Tree(object):
    """
    Decision Tree using Gini index for the splitting criterion.

    Parameters:
    -----------
    get_data: function (parameters: indices)
        Method to retrieve data.
    epsilon: float (default=0.1)
        Controls the level of indistinguishability; lower for stricter gaurantees,
        higher for more deletion efficiency.
    lmbda: float (default=0.1)
        Controls the amount of noise injected into the learning algorithm.
    gamma: float (default=0.1)
        Fraction of data to support for removal.
    max_depth: int (default=4)
        The maximum depth of a tree.
    min_samples_split: int (default=2)
        The minimum number of samples needed to make a split when building a tree.
    random_state: int (default=None)
        Random state for reproducibility.
    verbose: int (default=0)
        Verbosity level.
    feature_indices: list (default=None)
        Indices of the features present in this tree.
    """
    def __init__(self, epsilon=0.1, lmbda=0.1, gamma=0.1, max_depth=4, min_samples_split=2,
                 random_state=None, verbose=0, get_data=None, feature_indices=None):
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
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
        s += '\ngamma={}'.format(self.gamma)
        s += '\nmax_depth={}'.format(self.max_depth)
        s += '\nmin_samples_split={}'.format(self.min_samples_split)
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

        # save the data for easy deletion
        if self.single_tree_:
            self.X_train_, self.y_train_ = self._numpy_to_dict(X, y)
            keys = np.arange(X.shape[0])
        else:
            assert keys is not None

        self.root_ = self._build_tree(X, y, keys)
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

    def add(self, X, y, keys=None):
        """
        Adds instances to the training data and updates the model.
        """
        assert X.ndim == 2 and y.ndim == 1

        # assign index numbers to the new instances
        if self.single_tree_:
            current_keys = np.fromiter(self.X_train_.keys(), dtype=np.int64)
            gaps = np.setdiff1d(np.arange(current_keys.max()), current_keys)
            if len(X) > len(gaps):
                extra = np.arange(current_keys.max() + 1, current_keys.max() + 1 + len(X) - len(gaps))
            keys = np.concatenate([gaps, extra])
        else:
            X = X[:, self.feature_indices]
            assert keys is not None

        # add instances to the data
        if self.single_tree_:
            for i, key in enumerate(keys):
                self.X_train_[key] = X[i]
                self.y_train_[key] = y[i]

        # update model
        self.addition_types_ = []
        self.root_ = self._add(X, y, keys)

        return self.addition_types_

    def delete(self, remove_indices):
        """
        Removes instance remove_ndx from the training data and updates the model.
        """
        if isinstance(remove_indices, int):
            remove_indices = np.array([remove_indices], dtype=np.int32)

        # get data to remove
        X, y, keys = self.get_data(remove_indices)
        if not self.single_tree_:
            X = X[:, self.feature_indices]

        # update model
        self.deletion_types_ = []
        self.root_ = self._delete(X, y, remove_indices)

        # remove the instances from the data
        if self.single_tree_:
            for remove_ndx in remove_indices:
                del self.X_train_[remove_ndx]
                del self.y_train_[remove_ndx]

        return self.deletion_types_

    # private
    def _build_tree(self, X, y, keys, current_depth=0):
        """
        Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data.
        """

        # additional data structure to maintain attribute split info
        n_samples = len(keys)
        pos_count = np.sum(y)
        neg_count = n_samples - pos_count
        gini_data = round(1 - (pos_count / n_samples)**2 - (neg_count / n_samples)**2, 8)
        node_dict = {'count': n_samples, 'pos_count': pos_count, 'gini_data': gini_data}

        # handle edge cases
        create_leaf = False
        if node_dict['count'] > 0:

            # all instances of the same class
            if node_dict['pos_count'] == 0 or node_dict['pos_count'] == node_dict['count']:

                # the root node contains instances from the same class
                if current_depth == 0:
                    raise ValueError('root node contains only instances from the same class!')

                # create leaf
                else:
                    create_leaf = True

        else:
            raise ValueError('Zero samples in this node!, depth: {}'.format(current_depth))

        # create leaf
        if n_samples < self.min_samples_split or current_depth == self.max_depth or \
                self.n_features_ - current_depth == 0 or create_leaf:
            leaf_value = pos_count / n_samples
            node_dict['count'] = n_samples
            node_dict['pos_count'] = pos_count
            node_dict['leaf_value'] = 0 if node_dict['pos_count'] == 0 else node_dict['pos_count'] / node_dict['count']
            node_dict['indices'] = keys
            return DecisionNode(value=leaf_value, node_dict=node_dict)

        # create a decision node
        else:

            # save gini indexes from each attribute
            gini_indexes = []
            attr_indices = []

            # iterate through each feature
            node_dict['attr'] = {}
            for i in range(self.n_features_):

                # split the binary attribute
                left_indices = np.where(X[:, i] == 1)[0]
                right_indices = np.setdiff1d(np.arange(n_samples), left_indices)

                # debug
                if self.verbose > 1:
                    print(i, y[left_indices].shape, y[right_indices].shape, current_depth)

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

                    # save the metadata for efficient updating
                    node_dict['attr'][i] = {'left': {}, 'right': {}}
                    node_dict['attr'][i]['left']['count'] = left_count
                    node_dict['attr'][i]['left']['pos_count'] = left_pos_count
                    node_dict['attr'][i]['left']['weight'] = left_weight
                    node_dict['attr'][i]['left']['pos_prob'] = left_pos_prob
                    node_dict['attr'][i]['left']['index'] = left_index
                    node_dict['attr'][i]['left']['weighted_index'] = left_weighted_index

                    node_dict['attr'][i]['right']['count'] = right_count
                    node_dict['attr'][i]['right']['pos_count'] = right_pos_count
                    node_dict['attr'][i]['right']['weight'] = right_weight
                    node_dict['attr'][i]['right']['pos_prob'] = right_pos_prob
                    node_dict['attr'][i]['right']['index'] = right_index
                    node_dict['attr'][i]['right']['weighted_index'] = right_weighted_index

                    gini_index = self._compute_gini_index(node_dict['attr'][i])
                    node_dict['attr'][i]['gini_index'] = gini_index

                    # save gini_indexes for later
                    gini_indexes.append(gini_index)
                    attr_indices.append(i)

            # all attributes create hanging branches - create leaf
            if len(gini_indexes) == 0:
                leaf_value = pos_count / n_samples
                node_dict['count'] = n_samples
                node_dict['pos_count'] = pos_count
                node_dict['leaf_value'] = 0 if pos_count == 0 else pos_count / n_samples
                node_dict['indices'] = keys
                return DecisionNode(value=leaf_value, node_dict=node_dict)

            # create probability distribution over the attributes
            p = self._generate_distribution(gini_indexes)

            # sample from this distribution
            np.random.seed(self.random_state)
            chosen_i = np.random.choice(attr_indices, p=p)

            # retrieve samples for the chosen attribute
            left_indices = np.where(X[:, chosen_i] == 1)[0]
            right_indices = np.setdiff1d(np.arange(n_samples), left_indices)

            # build the node with the chosen attribute
            left = self._build_tree(X[left_indices], y[left_indices], keys[left_indices], current_depth + 1)
            right = self._build_tree(X[right_indices], y[right_indices], keys[right_indices], current_depth + 1)
            return DecisionNode(feature_i=chosen_i, node_dict=node_dict, left_branch=left, right_branch=right)

    def _generate_distribution(self, gini_indexes, invalid_indices=[], cur_ndx=None):
        """
        Creates a probability distribution over the attributes given
        their gini index scores.
        """
        gini_indexes = np.array(gini_indexes)

        # numbers are too small, go into deterministic mode
        if np.exp(-(self.lmbda * gini_indexes.min()) / (5 * self.gamma)) == 0:
            p = np.where(gini_indexes == gini_indexes.min(), 1, 0)

            # there is a tie between 2 or more attributes
            if p.sum() > 1:

                # choose existing attribute if it is tied for first
                if cur_ndx and p[cur_ndx] == 1:
                    chosen_ndx = cur_ndx

                # choose the first attribute encountered that is tied for first
                else:
                    chosen_ndx = np.argmax(p == 1)

                # set the probability of the other attributes to zero
                p[np.setdiff1d(np.arange(len(p)), chosen_ndx)] = 0

        # create probability distribution over the attributes
        else:
            p = np.exp(-(self.lmbda * gini_indexes) / (5 * self.gamma))
            if len(invalid_indices) > 0:
                p[np.array(invalid_indices)] = 0
            p = p / p.sum()

        return p

    def _add(self, X, y, add_indices, tree=None, current_depth=0):

        # get root node of the tree
        if tree is None:
            tree = self.root_

        # type 1: leaf node, update its metadata
        if tree.value is not None:
            self._increment_leaf_node(tree, y, add_indices)

            if self.verbose > 0:
                print('tree check complete, ended at depth {}'.format(current_depth))

            self.addition_types_.append('1')
            return tree

        # decision node, update the high-level metadata
        count = tree.node_dict['count'] + len(y)
        pos_count = tree.node_dict['pos_count'] + np.sum(y)
        neg_count = pos_count - count
        gini_data = round(1 - (pos_count / count)**2 - (neg_count / count)**2, 8)
        tree.node_dict['pos_count'] = pos_count
        tree.node_dict['count'] = count
        tree.node_dict['gini_data'] = gini_data

        # udpate gini_index for each attribute in this node
        old_gini_indexes = []
        gini_indexes = []

        for i, attr_ndx in enumerate(tree.node_dict['attr']):

            left_indices = np.where(X[:, attr_ndx] == 1)[0]
            right_indices = np.setdiff1d(np.arange(X.shape[0]), left_indices)
            y_left, y_right = y[left_indices], y[right_indices]

            if len(y_left) > 0:
                self._increment_decision_node(tree.node_dict, attr_ndx, 'left', y_left)

            if len(y_right) > 0:
                self._increment_decision_node(tree.node_dict, attr_ndx, 'right', y_right)

            # recompute the gini index for this attribute
            attr_dict = tree.node_dict['attr'][attr_ndx]
            gini_index = self._compute_gini_index(attr_dict)
            old_gini_indexes.append(attr_dict['gini_index'])
            gini_indexes.append(gini_index)
            attr_dict['gini_index'] = gini_index

        # get old and updated probability distributions
        old_p = self._generate_distribution(old_gini_indexes)
        p = self._generate_distribution(gini_indexes, cur_ndx=np.argmax(old_p))

        # retrain if probability ratio over any attribute differs by more than e^ep or e^-ep
        if np.any(p / old_p > np.exp(self.epsilon)) or np.any(p / old_p < np.exp(-self.epsilon)):

            if self.verbose > 0:
                print('rebuilding at depth {}'.format(current_depth))

            indices = self._get_indices(tree, current_depth)
            indices = self._add_elements(indices, add_indices)
            Xa, ya, keys = self.get_data(indices)
            self.deletion_types_.append('{}_{}'.format('2', current_depth))

            return self._build_tree(Xa, ya, keys, current_depth)

        # continue checking the tree
        else:

            left_indices = np.where(X[:, tree.feature_i] == 1)[0]
            right_indices = np.setdiff1d(np.arange(X.shape[0]), left_indices)
            y_left, y_right = y[left_indices], y[right_indices]

            if len(left_indices) > 0:

                if self.verbose > 0:
                    print('check complete at depth {}, traversing left'.format(current_depth))

                X_left = X[left_indices]
                left_add_indices = add_indices[left_indices]
                left_branch = self._add(X_left, y_left, left_add_indices, tree=tree.left_branch,
                                        current_depth=current_depth + 1)
                tree.left_branch = left_branch

            if len(right_indices) > 0:

                if self.verbose > 0:
                    print('check complete at depth {}, traversing right'.format(current_depth))

                X_right = X[right_indices]
                right_add_indices = add_indices[right_indices]
                right_branch = self._add(X_right, y_right, right_add_indices, tree=tree.right_branch,
                                         current_depth=current_depth + 1)
                tree.right_branch = right_branch

            return tree

    def _delete(self, X, y, remove_indices, tree=None, current_depth=0):

        # get root node of the tree
        if tree is None:
            tree = self.root_

        # type 1: leaf node, update its metadata
        if tree.value is not None:
            self._decrement_leaf_node(tree, y, remove_indices)

            if self.verbose > 0:
                print('tree check complete, ended at depth {}'.format(current_depth))

            self.deletion_types_.append('1a')
            return tree

        # decision node, update the high-level metadata
        count = tree.node_dict['count'] - len(y)
        pos_count = tree.node_dict['pos_count'] - np.sum(y)
        neg_count = pos_count - count
        gini_data = round(1 - (pos_count / count)**2 - (neg_count / count)**2, 8)
        tree.node_dict['pos_count'] = pos_count
        tree.node_dict['count'] = count
        tree.node_dict['gini_data'] = gini_data

        # raise an error if there are only instances from one class are at the root
        if current_depth == 0:
            if tree.node_dict['pos_count'] == 0 or tree.node_dict['pos_count'] == tree.node_dict['count']:
                raise ValueError('Instances in the root node are all from the same class!')

        # edge case: if remaining instances in this node are of the same class, make leaf
        if tree.node_dict['pos_count'] == 0 or tree.node_dict['pos_count'] == tree.node_dict['count']:

            if self.verbose > 0:
                print('check complete, lefotvers in the same class, creating leaf at depth {}'.format(current_depth))

            tree.node_dict['attr'] = None
            tree.node_dict['leaf_value'] = tree.node_dict['pos_count'] / tree.node_dict['count']
            tree.node_dict['indices'] = self._get_indices(tree, current_depth)
            tree.node_dict['indices'] = self._remove_elements(tree.node_dict['indices'], remove_indices)
            tree_branch = DecisionNode(value=tree.node_dict['leaf_value'], node_dict=tree.node_dict)
            self.deletion_types_.append('1b_{}'.format(current_depth))
            return tree_branch

        # type 2: all instances are removed from the left or right branch, rebuild at this node
        # udpate gini_index for each attribute in this node
        old_gini_indexes = []
        gini_indexes = []
        invalid_indices = []
        invalid_attr_indices = []

        for i, attr_ndx in enumerate(tree.node_dict['attr']):
            left_status, right_status = True, True

            left_indices = np.where(X[:, attr_ndx] == 1)[0]
            right_indices = np.setdiff1d(np.arange(X.shape[0]), left_indices)
            y_left, y_right = y[left_indices], y[right_indices]

            if len(y_left) > 0:
                left_status = self._decrement_decision_node(tree.node_dict, attr_ndx, 'left', y_left)

            if len(y_right) > 0:
                right_status = self._decrement_decision_node(tree.node_dict, attr_ndx, 'right', y_right)

            # this attribute causes a hanging branch, remove it from future tree models
            if left_status is None or right_status is None:
                invalid_attr_indices.append(attr_ndx)
                invalid_indices.append(i)
                old_gini_indexes.append(tree.node_dict['attr'][attr_ndx]['gini_index'])
                gini_indexes.append(1)

            # recompute the gini index for this attribute
            else:
                attr_dict = tree.node_dict['attr'][attr_ndx]
                gini_index = self._compute_gini_index(attr_dict)
                old_gini_indexes.append(attr_dict['gini_index'])
                gini_indexes.append(gini_index)
                attr_dict['gini_index'] = gini_index

        # remove invalid attributes from the model
        for invalid_attr_ndx in invalid_attr_indices:
            del tree.node_dict['attr'][invalid_attr_ndx]

        # get old and updated probability distributions
        old_p = self._generate_distribution(old_gini_indexes)
        p = self._generate_distribution(gini_indexes, invalid_indices=invalid_indices, cur_ndx=np.argmax(old_p))

        # retrain if probability ratio over any attribute differs by more than e^ep or e^-ep
        if np.any(p / old_p > np.exp(self.epsilon)) or np.any(p / old_p < np.exp(-self.epsilon)):

            if self.verbose > 0:
                print('rebuilding at depth {}'.format(current_depth))

            indices = self._get_indices(tree, current_depth)
            indices = self._remove_elements(indices, remove_indices)
            Xa, ya, keys = self.get_data(indices)

            dtype = '2a' if len(invalid_indices) > 0 else '2b'
            self.deletion_types_.append('{}_{}'.format(dtype, current_depth))

            return self._build_tree(Xa, ya, keys, current_depth)

        # continue checking the tree
        else:

            left_indices = np.where(X[:, tree.feature_i] == 1)[0]
            right_indices = np.setdiff1d(np.arange(X.shape[0]), left_indices)
            y_left, y_right = y[left_indices], y[right_indices]

            if len(left_indices) > 0:

                if self.verbose > 0:
                    print('check complete at depth {}, traversing left'.format(current_depth))

                X_left = X[left_indices]
                left_remove_indices = remove_indices[left_indices]
                left_branch = self._delete(X_left, y_left, left_remove_indices, tree=tree.left_branch,
                                           current_depth=current_depth + 1)
                tree.left_branch = left_branch

            if len(right_indices) > 0:

                if self.verbose > 0:
                    print('check complete at depth {}, traversing right'.format(current_depth))

                X_right = X[right_indices]
                right_remove_indices = remove_indices[right_indices]
                right_branch = self._delete(X_right, y_right, right_remove_indices, tree=tree.right_branch,
                                            current_depth=current_depth + 1)
                tree.right_branch = right_branch

            return tree

    def _increment_leaf_node(self, tree, y, add_indices):
        """
        Update this leaf node to effectively add the target indices.
        """
        node_dict = tree.node_dict
        node_dict['count'] += len(y)
        node_dict['pos_count'] += np.sum(y)
        node_dict['leaf_value'] = 0 if node_dict['pos_count'] == 0 else node_dict['pos_count'] / node_dict['count']
        node_dict['indices'] = self._add_elements(node_dict['indices'], add_indices)
        tree.value = node_dict['leaf_value']

    def _decrement_leaf_node(self, tree, y, remove_indices):
        """
        Update this leaf node to effectively remove the target indices.
        """
        node_dict = tree.node_dict
        node_dict['count'] -= len(y)
        node_dict['pos_count'] -= np.sum(y)
        node_dict['leaf_value'] = 0 if node_dict['pos_count'] == 0 else node_dict['pos_count'] / node_dict['count']
        node_dict['indices'] = self._remove_elements(node_dict['indices'], remove_indices)
        tree.value = node_dict['leaf_value']

    def _increment_decision_node(self, node_dict, attr_ndx, abranch, y):
        """
        Update the attribute dictionary of the node metadata of an addition operation.
        """

        # access the attriubute metadata
        abranch_dict = node_dict['attr'][attr_ndx][abranch]

        # update the affected branch
        abranch_dict['count'] += len(y)
        abranch_dict['pos_count'] += np.sum(y)
        abranch_dict['weight'] = abranch_dict['count'] / node_dict['count']
        abranch_dict['pos_prob'] = abranch_dict['pos_count'] / abranch_dict['count']
        abranch_dict['index'] = 1 - (np.square(abranch_dict['pos_prob']) + np.square(1 - abranch_dict['pos_prob']))
        abranch_dict['weighted_index'] = abranch_dict['weight'] * abranch_dict['index']

        return True

    def _decrement_decision_node(self, node_dict, attr_ndx, abranch, y):
        """
        Update the attribute dictionary of the node metadata from a deletion operation.
        """

        # access the attriubute metadata
        abranch_dict = node_dict['attr'][attr_ndx][abranch]

        # # only the affected instances are in this branch
        if abranch_dict['count'] <= len(y):
            return None

        # update the affected branch
        abranch_dict['count'] -= len(y)
        abranch_dict['pos_count'] -= np.sum(y)
        abranch_dict['weight'] = abranch_dict['count'] / node_dict['count']
        abranch_dict['pos_prob'] = abranch_dict['pos_count'] / abranch_dict['count']
        abranch_dict['index'] = 1 - (np.square(abranch_dict['pos_prob']) + np.square(1 - abranch_dict['pos_prob']))
        abranch_dict['weighted_index'] = abranch_dict['weight'] * abranch_dict['index']

        return True

    def _compute_gini_index(self, attr_dict):
        gini_index = attr_dict['left']['weighted_index'] + attr_dict['right']['weighted_index']
        return round(gini_index, 8)

    def _get_indices(self, tree=None, depth=0):
        """
        Recursively retrieve all the indices for this node from the leaves.
        """
        if tree is None:
            tree = self.root_

        # made it to a leaf node, return the indices
        if tree.value is not None:
            return tree.node_dict['indices']

        else:
            left_indices = self._get_indices(tree.left_branch, depth + 1)
            right_indices = self._get_indices(tree.right_branch, depth + 1)
            return np.concatenate([left_indices, right_indices])

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

    def _add_elements(self, arr, elements):
        """
        Add elements to array.
        """
        return np.concatenate([arr, elements])

    def _remove_elements(self, arr, elements):
        """
        Remove elements from array.
        """
        return np.setdiff1d(arr, elements)

    def _get_numpy_data(self, indices):
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
    left_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    right_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    node_dict: dict
        Dictionary to store metatadata for easy updating.
    """
    def __init__(self, feature_i=None, threshold=None, value=None, left_branch=None, right_branch=None,
                 node_dict=None):
        self.feature_i = feature_i          # Index for the feature that is tested
        self.threshold = threshold          # Threshold value for feature
        self.value = value                  # Value if the node is a leaf in the tree
        self.left_branch = left_branch      # 'Left' subtree
        self.right_branch = right_branch    # 'Right' subtree
        self.node_dict = node_dict          # Attribute split / leaf metadata

    def copy(self):
        left_node = self.left_branch.copy() if self.left_branch is not None else None
        right_node = self.right_branch.copy() if self.right_branch is not None else None
        node_dict = self.node_dict.copy()
        node = DecisionNode(feature_i=self.feature_i, threshold=self.threshold, value=self.value,
                            left_branch=left_node, right_branch=right_node, node_dict=node_dict)
        return node
