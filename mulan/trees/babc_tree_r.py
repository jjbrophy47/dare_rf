"""
Decision tree implementation for binary classification with binary attributes.
Adapted from MLFromScratch: https://github.com/eriklindernoren/ML-From-Scratch.
"""
import numpy as np


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


class BABC_Tree_R(object):
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
    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=4, alpha=0.01, verbose=0, seed=None):
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.alpha = alpha
        self.verbose = verbose
        self.seed = seed

    def __str__(self):
        s = 'Tree:'
        s += '\nmin_samples_split={}'.format(self.min_samples_split)
        s += '\nmin_impurity={}'.format(self.min_impurity)
        s += '\nmax_depth={}'.format(self.max_depth)
        s += '\nverbose={}'.format(self.verbose)
        s += '\nseed={}'.format(self.seed)
        return s

    def fit(self, X, y):
        """
        Build decision tree.
        """
        assert X.ndim == 2
        assert y.ndim == 1
        self.n_features_ = X.shape[1]

        # save the data into dicts for easy deletion
        self.X_train_, self.y_train_ = self._numpy_to_dict(X, y)
        keys = np.arange(X.shape[0])

        self.root_ = self._build_tree(X, y, keys)
        return self

    def _remove_element(self, arr, element):
        return arr[arr != element]

    def _numpy_to_dict(self, X, y):
        """
        Converts numpy data into dicts.
        """
        Xd, yd = {}, {}
        for i in range(X.shape[0]):
            Xd[i] = X[i]
            yd[i] = y[i]
        return Xd, yd

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

    def _build_tree(self, X, y, keys, current_depth=0):
        """
        Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data.
        """

        # keeps track of the best attribute
        best_gini_gain = -1

        # additional data structure to maintain attribute split info
        n_samples = len(keys)
        pos_count = np.sum(y)
        neg_count = n_samples - pos_count
        gini_data = round(1 - (pos_count / n_samples)**2 - (neg_count / n_samples)**2, 8)
        node_dict = {'count': n_samples, 'pos_count': pos_count, 'gini_data': gini_data}

        # to support randomization of attribute picking
        attr_pool = {}

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

        # error checking
        if n_samples >= self.min_samples_split and current_depth < self.max_depth and \
                self.n_features_ - current_depth > 0 and not create_leaf:
            node_dict['attr'] = {}

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
                    gini_gain = gini_data - gini_index
                    node_dict['attr'][i]['gini_index'] = gini_index
                    node_dict['attr'][i]['gini_gain'] = gini_gain

                    # keep the best attribute
                    if gini_gain > best_gini_gain:
                        best_gini_gain = gini_gain

                        # make sure all attributes in the pool are close in gini gain
                        keys_to_delete = []
                        for k in attr_pool.keys():
                            if best_gini_gain - attr_pool[k]['gini_gain'] > self.alpha:
                                keys_to_delete.append(k)

                        for k in keys_to_delete:
                            del attr_pool[k]

                        # add the new best attribute to the pool
                        attr_pool[i] = {'gini_gain': gini_gain, 'left_indices': left_indices,
                                        'right_indices': right_indices}

            # choose an attribute from a weighted distribution of the best attributes
            if len(attr_pool) > 0:

                if len(attr_pool) == 1:
                    chosen_feature_ndx = list(attr_pool.keys())[0]
                else:
                    attribute_choices = list(attr_pool.keys())
                    attribute_weights = np.array([attr_pool[k]['gini_gain'] for k in attribute_choices])
                    p = attribute_weights / attribute_weights.sum() if attribute_weights.sum() > 0 else None
                    np.random.seed(self.seed)
                    chosen_feature_ndx = np.random.choice(attribute_choices, p=p, size=1)[0]

                    if self.verbose > 0:
                        print(attribute_choices, p, chosen_feature_ndx)

                chosen_left_indices = attr_pool[chosen_feature_ndx]['left_indices']
                chosen_right_indices = attr_pool[chosen_feature_ndx]['right_indices']

                left_node = self._build_tree(X[chosen_left_indices], y[chosen_left_indices],
                                             keys[chosen_left_indices], current_depth + 1)
                right_node = self._build_tree(X[chosen_right_indices], y[chosen_right_indices],
                                              keys[chosen_right_indices], current_depth + 1)
                return DecisionNode(feature_i=chosen_feature_ndx, node_dict=node_dict,
                                    left_branch=left_node, right_branch=right_node)

        # we're at a leaf => determine value
        leaf_value = pos_count / n_samples
        node_dict['count'] = n_samples
        node_dict['pos_count'] = pos_count
        node_dict['leaf_value'] = 0 if node_dict['pos_count'] == 0 else node_dict['pos_count'] / node_dict['count']
        node_dict['indices'] = keys
        return DecisionNode(value=leaf_value, node_dict=node_dict)

    def copy(self):
        """
        Return a deep copy of this object.
        """
        tree = BABC_Tree_R(min_samples_split=self.min_samples_split, min_impurity=self.min_impurity,
                           max_depth=self.max_depth, verbose=self.verbose)
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
        y_positive = np.array([self._predict_value(sample) for sample in X]).reshape(-1, 1)
        y_pred = np.hstack([1 - y_positive, y_positive])
        return y_pred

    def delete(self, remove_ndx):
        """
        Removes instance remove_ndx from the training data and updates the model.
        """
        assert isinstance(remove_ndx, int)
        x = self.X_train_.get(remove_ndx)
        y = self.y_train_.get(remove_ndx)
        assert x is not None and y is not None

        self.deletion_type_ = None
        self.root_ = self._delete(x, y, remove_ndx)

        # remove the instance from the data
        del self.X_train_[remove_ndx]
        del self.y_train_[remove_ndx]

        return self.deletion_type_

    def _delete(self, x, y, remove_ndx, tree=None, current_depth=0):

        # get root node of the tree
        if tree is None:
            tree = self.root_

        # type 1a: leaf node, update its metadata
        if tree.value is not None:
            self._update_leaf_node(tree, remove_ndx)
            if self.verbose > 0:
                print('tree check complete, ended at depth {}'.format(current_depth))
            self.deletion_type_ = '1a'
            return tree

        # decision node, update the high-level metadata
        count = tree.node_dict['count'] - 1
        pos_count = tree.node_dict['pos_count'] - self.y_train_[remove_ndx]
        neg_count = pos_count - count
        gini_data = round(1 - (pos_count / count)**2 - (neg_count / count)**2, 8)
        tree.node_dict['pos_count'] = pos_count
        tree.node_dict['count'] = count
        tree.node_dict['gini_data'] = gini_data

        # find the affected branch
        abranch = 'left' if x[tree.feature_i] == 1 else 'right'

        # raise an error if there are only instances from one class are at the root
        if current_depth == 0:
            if tree.node_dict['pos_count'] == 0 or tree.node_dict['pos_count'] == tree.node_dict['count']:
                raise ValueError('Instances in the root node are all from the same class!')

        # type 1b: if remaining instances in this node are of the same class, make leaf
        if tree.node_dict['pos_count'] == 0 or tree.node_dict['pos_count'] == tree.node_dict['count']:
            if self.verbose > 0:
                pstr = 'tree check complete, remaining instances in the same class, creating a leaf at depth {}'
                print(pstr.format(current_depth))
            tree.node_dict['attr'] = None
            tree.node_dict['leaf_value'] = tree.node_dict['pos_count'] / tree.node_dict['count']
            tree.node_dict['indices'] = self.get_indices(tree, current_depth)
            tree.node_dict['indices'] = self._remove_element(tree.node_dict['indices'], remove_ndx)
            tree_branch = DecisionNode(value=tree.node_dict['leaf_value'], node_dict=tree.node_dict)
            self.deletion_type_ = '1b'
            return tree_branch

        # type 2: the affected branch only contains the instance to be removed
        if tree.node_dict['attr'][tree.feature_i][abranch]['count'] == 1:
            nabranch = 'right' if abranch == 'left' else 'left'
            na_branch = tree.left_branch if nabranch == 'left' else tree.right_branch

            # type 2a: both branches contain one example, turn this node into a leaf
            if tree.node_dict['attr'][tree.feature_i][nabranch]['count'] == 1:
                if self.verbose > 0:
                    print('tree check complete, creating a leaf at depth {}'.format(current_depth))
                tree.node_dict['attr'] = None
                tree.node_dict['leaf_value'] = tree.node_dict['pos_count'] / tree.node_dict['count']
                tree.node_dict['indices'] = self.get_indices(na_branch, current_depth + 1)
                tree.node_dict['indices'] = self._remove_element(tree.node_dict['indices'], remove_ndx)
                tree_branch = DecisionNode(value=tree.node_dict['leaf_value'], node_dict=tree.node_dict)
                self.deletion_type_ = '2a'
                return tree_branch

            # type 2b: other branch still has more than one instance, rebuild at this node
            else:
                if self.verbose > 0:
                    print('hanging branch with >1 instance, rebuilding at depth {}'.format(current_depth))
                indices = self.get_indices(tree, current_depth)
                indices = self._remove_element(indices, remove_ndx)
                Xa, ya, keys = self._get_numpy_data(indices)
                self.deletion_type_ = '2b_{}'.format(current_depth)
                return self._build_tree(Xa, ya, keys, current_depth)

        # check if other attributes are better than the current attribute
        attr_dict = tree.node_dict['attr']
        rebuild = False

        # update the metadata for the current attribute
        if self._update_decision_node(tree.node_dict, tree.feature_i, x, y) is None:
            rebuild = True

        # update metadata for other attributes and see if any are better by alpha
        if not rebuild:
            for attr_ndx in tree.node_dict['attr']:
                if attr_ndx == tree.feature_i or self._update_decision_node(tree.node_dict, attr_ndx, x, y) is None:
                    continue

                if attr_dict[attr_ndx]['gini_gain'] - attr_dict[tree.feature_i]['gini_gain'] > self.alpha:
                    rebuild = True
                    break

        # traverse the affected branch and continue checking
        if not rebuild:

            if self.verbose > 0:
                print('check complete at depth {}, traversing {}'.format(current_depth, abranch))

            tree_branch = tree.left_branch if abranch == 'left' else tree.right_branch
            new_branch = self._delete(x, y, remove_ndx, tree=tree_branch, current_depth=current_depth + 1)

            if abranch == 'left':
                tree.left_branch = new_branch
            else:
                tree.right_branch = new_branch

            return tree

        # type 3: the attribute has changed, rebuild at this node
        else:
            if self.verbose > 0:
                print('rebuilding at depth {}'.format(current_depth))
            indices = self.get_indices(tree, current_depth)
            indices = self._remove_element(indices, remove_ndx)
            Xa, ya, keys = self._get_numpy_data(indices)
            self.deletion_type_ = '3_{}'.format(current_depth)
            return self._build_tree(Xa, ya, keys, current_depth)

    def _update_leaf_node(self, tree, remove_ndx):
        """
        Update this leaf node to effectively remove the target index.
        """
        node_dict = tree.node_dict
        node_dict['pos_count'] -= self.y_train_[remove_ndx]
        node_dict['count'] -= 1
        node_dict['leaf_value'] = 0 if node_dict['pos_count'] == 0 else node_dict['pos_count'] / node_dict['count']
        node_dict['indices'] = self._remove_element(node_dict['indices'], remove_ndx)
        tree.value = node_dict['leaf_value']

    def _update_decision_node(self, node_dict, attr_ndx, x, y_val):
        """
        Update the attribute dictionary of the node metadata.
        """

        # access the attribute metadata
        abranch = 'left' if x[attr_ndx] == 1 else 'right'
        abranch_dict = node_dict['attr'][attr_ndx][abranch]

        # only the affected instance is in this branch
        if abranch_dict['count'] <= 1:
            return None

        # update the affected branch
        abranch_dict['count'] -= 1
        abranch_dict['pos_count'] -= y_val
        abranch_dict['weight'] = abranch_dict['count'] / node_dict['count']
        abranch_dict['pos_prob'] = abranch_dict['pos_count'] / abranch_dict['count']
        abranch_dict['index'] = 1 - (np.square(abranch_dict['pos_prob']) + np.square(1 - abranch_dict['pos_prob']))
        abranch_dict['weighted_index'] = abranch_dict['weight'] * abranch_dict['index']

        # update the non-affected branch
        nabranch = 'left' if abranch == 'right' else 'right'
        nabranch_dict = node_dict['attr'][attr_ndx][nabranch]
        nabranch_dict['weight'] = nabranch_dict['count'] / node_dict['count']
        nabranch_dict['weighted_index'] = nabranch_dict['weight'] * nabranch_dict['index']

        attr_dict = node_dict['attr'][attr_ndx]
        gini_index = self._compute_gini_index(attr_dict)
        gini_gain = node_dict['gini_data'] - gini_index
        attr_dict['gini_index'] = gini_index
        attr_dict['gini_gain'] = gini_gain

        return 1

    def _compute_gini_index(self, attr_dict):
        gini_index = attr_dict['left']['weighted_index'] + attr_dict['right']['weighted_index']
        return round(gini_index, 8)

    def get_indices(self, tree=None, depth=0):
        """
        Recursively retrieve all the indices for this node from the leaves.
        """
        if tree is None:
            tree = self.root_

        # made it to a leaf node, return the indices
        if tree.value is not None:
            return tree.node_dict['indices']

        else:
            left_indices = self.get_indices(tree.left_branch, depth + 1)
            right_indices = self.get_indices(tree.right_branch, depth + 1)
            return np.concatenate([left_indices, right_indices])

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

    def _predict_value(self, x, tree=None):
        """
        Do a recursive search down the tree and make a prediction of the data sample by the
        value of the leaf that we end up at.
        """

        if tree is None:
            tree = self.root_

        # If we have a value (i.e we're at a leaf) => return value as the prediction
        if tree.value is not None:
            return tree.value

        # traverse the tree based on the attribute value
        if x[tree.feature_i] == 1:
            branch = tree.left_branch

        else:
            branch = tree.right_branch

        # test subtree
        return self._predict_value(x, branch)
