"""
Decision tree implementation for binary classification with binary attributes.
Uses numpy arrays to build the tree.
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


class BABC_Tree_Old(object):
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
    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=4, verbose=0):
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.verbose = verbose

    def __str__(self):
        s = 'Tree:'
        s += '\nmin_samples_split={}'.format(self.min_samples_split)
        s += '\nmin_impurity={}'.format(self.min_impurity)
        s += '\nmax_depth={}'.format(self.max_depth)
        return s

    def fit(self, X, y):
        """
        Build decision tree.
        """
        assert y.ndim == 1
        assert X.ndim == 2
        assert X.shape[0] == len(y)
        self.n_features_ = X.shape[1]
        self.X_train_ = X
        self.y_train_ = y
        self.root_ = self._build_tree(np.arange(len(X)))
        return self

    def _build_tree(self, indices, current_depth=0):
        """
        Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data.
        """

        # keeps track of the best attribute
        best_gini_index = 1e7
        best_feature_ndx = None

        # additional data structure to maintain attribute split info
        n_samples = len(indices)
        node_dict = {'count': n_samples, 'pos_count': np.sum(self.y_train_[indices])}

        # handle edge cases
        create_leaf = False
        if node_dict['count'] > 0:

            # all instances of the same class
            if node_dict['pos_count'] == 0 or node_dict['pos_count'] == node_dict['count']:

                # the root node contains instances ll from the same class
                if current_depth == 0:
                    raise ValueError('root node contains only instances from the same class!')

                # create leaf
                else:
                    create_leaf = True

        else:
            raise ValueError('Zero samples in this node!')

        # error checking
        if n_samples >= self.min_samples_split and current_depth < self.max_depth and \
                self.n_features_ - current_depth > 0 and not create_leaf:
            node_dict['attr'] = {}

            # iterate through each feature
            for i in range(self.n_features_):

                # split the binary attribute
                left_indices = indices[np.where(self.X_train_[indices][:, i] == 1)]
                right_indices = np.setdiff1d(indices, left_indices)

                if self.verbose > 1:
                    print(i, left_indices, right_indices)
                    print(i, self.y_train_[left_indices], self.y_train_[right_indices])

                # make sure there is atleast 1 sample in each branch
                if len(left_indices) > 0 and len(right_indices) > 0:

                    # print(left_indices, right_indices)

                    # gather stats about the split to compute the Gini index
                    left_count = len(left_indices)
                    left_pos_count = np.sum(self.y_train_[left_indices])
                    right_count = n_samples - left_count
                    right_pos_count = np.sum(self.y_train_[right_indices])

                    # print(i, left_count, left_pos_count, right_count, right_pos_count)

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

                    # if i == 0:
                    #     print(node_dict)
                    #     print(self.X_train_, self.y_train_)

                    # print(i, gini_index)
                    # exit(0)

                    # keep the best attribute
                    if gini_index < best_gini_index:
                        best_gini_index = gini_index
                        best_left_indices = left_indices
                        best_right_indices = right_indices
                        best_feature_ndx = i

            # split on the best saved attribute
            if best_feature_ndx is not None and best_gini_index >= 0:
                left_node = self._build_tree(best_left_indices, current_depth + 1)
                right_node = self._build_tree(best_right_indices, current_depth + 1)
                return DecisionNode(feature_i=best_feature_ndx, node_dict=node_dict,
                                    left_branch=left_node, right_branch=right_node)

        # we're at a leaf => determine value
        leaf_value = np.mean(self.y_train_[indices])
        node_dict['pos_count'] = np.sum(self.y_train_[indices])
        node_dict['count'] = len(indices)
        node_dict['indices'] = indices
        return DecisionNode(value=leaf_value, node_dict=node_dict)

    def predict(self, X):
        """
        Classify samples one by one and return the set of labels.
        """
        assert X.ndim == 2
        y_pred = [self._predict_value(sample) for sample in X]
        return y_pred

    def delete(self, remove_ndx):
        """
        Removes instance remove_ndx from the training data and updates the model.
        """
        assert isinstance(remove_ndx, int)
        assert remove_ndx <= self.X_train_.shape[0]
        x = self.X_train_[remove_ndx]
        y = self.y_train_[remove_ndx]
        self.deletion_type_ = None
        self.root_ = self._delete(x, y, remove_ndx)

        # remove the instance from the data
        self.X_train_ = np.delete(self.X_train_, remove_ndx, axis=0)
        self.y_train_ = np.delete(self.y_train_, remove_ndx)

        # update all leaf indices
        self._update_leaves(self.root_, remove_ndx)

        return self.deletion_type_

    def _delete(self, x, y, remove_ndx, tree=None, current_depth=0):

        # get root node of the tree
        if tree is None:
            tree = self.root_

        # type 1: leaf node, update its metadata
        if tree.value is not None:
            self._update_leaf_node(tree, remove_ndx)
            if self.verbose > 0:
                print('check complete, ended at depth {}'.format(current_depth))
            self.deletion_type_ = '1a'
            return tree

        # decision node, update the high-level metadata
        tree.node_dict['pos_count'] -= self.y_train_[remove_ndx]
        tree.node_dict['count'] -= 1

        # find the affected branch
        abranch = 'left' if x[tree.feature_i] == 1 else 'right'

        # raise an error if there are only instances from one class at the root
        if current_depth == 0:
            if tree.node_dict['pos_count'] == 0 or tree.node_dict['pos_count'] == tree.node_dict['count']:
                raise ValueError('Instances in the root node are all from the same class!')

        # edge case: if remaining instances in this node are of the same class, make leaf
        if tree.node_dict['pos_count'] == 0 or tree.node_dict['pos_count'] == tree.node_dict['count']:
            if self.verbose > 0:
                pstr = 'tree check complete, remaining instances in the same class, creating a leaf at depth {}'
                print(pstr.format(current_depth))
            tree.node_dict['attr'] = None
            tree.node_dict['leaf_value'] = tree.node_dict['pos_count'] / tree.node_dict['count']
            tree.node_dict['indices'] = self.get_indices(tree, current_depth)
            tree.node_dict['indices'] = tree.node_dict['indices'][tree.node_dict['indices'] != remove_ndx]
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
                tree.node_dict['indices'] = tree.node_dict['indices'][tree.node_dict['indices'] != remove_ndx]
                tree_branch = DecisionNode(value=tree.node_dict['leaf_value'], node_dict=tree.node_dict)
                self.deletion_type_ = '2a'
                return tree_branch

            # type 2b: other branch still has more than one instance, rebuild at this node
            else:
                if self.verbose > 0:
                    print('hanging branch with >1 instance, rebuilding at depth {}'.format(current_depth))
                indices = self.get_indices(tree, current_depth)
                indices = indices[indices != remove_ndx]
                self.deletion_type_ = '2b'
                return self._build_tree(indices, current_depth)

        # keep track of the new best attribute
        best_attr_ndx = None
        best_gini_index = 1e7

        # print(tree.node_dict)

        # update the metadata for each attribute's affected branch
        for attr_ndx in tree.node_dict['attr']:
            abranch = 'left' if x[attr_ndx] == 1 else 'right'
            if self._update_decision_node(tree.node_dict, attr_ndx, abranch, y) is None:
                continue

            # if self._update_decision_node(attr_dict[abranch], y, tree.node_dict['count']) is None:
            #     continue

            # recompute the gini index for this attribute
            attr_dict = tree.node_dict['attr'][attr_ndx]
            gini_index = self._compute_gini_index(attr_dict)
            attr_dict['gini_index'] = gini_index
            # print(attr_ndx, gini_index, attr_dict)

            # print(attr_dict, attr_ndx, gini_index)

            # save the attribute with the best gini index
            if gini_index < best_gini_index:
                best_gini_index = gini_index
                best_attr_ndx = attr_ndx
                best_branch = abranch

        # print(tree.node_dict)

        # edge case: keep original attribute split if it is tied for the smallest error after removal
        # no long useful because original building just picks the first best attribute
        # if tree.node_dict['attr'][tree.feature_i]['gini_index'] == best_gini_index:
        #     best_attr_ndx = tree.feature_i
        #     best_branch = 'left' if x[best_attr_ndx] == 1 else 'right'

        # check to see if the tree needs to be restructured
        if best_attr_ndx == tree.feature_i:

            # traverse the affected branch and continue checking
            if self.verbose > 0:
                print('check complete at depth {}, traversing {}'.format(current_depth, best_branch))
            tree_branch = tree.left_branch if best_branch == 'left' else tree.right_branch
            new_branch = self._delete(x, y, remove_ndx, tree=tree_branch, current_depth=current_depth + 1)

            if best_branch == 'left':
                tree.left_branch = new_branch
            else:
                tree.right_branch = new_branch
            return tree

        # type 3: split data left and right, rebuild at this node
        else:
            if self.verbose > 0:
                print('rebuilding at depth {}'.format(current_depth))
            indices = self.get_indices(tree, current_depth)
            indices = indices[indices != remove_ndx]
            self.deletion_type_ = '3'
            return self._build_tree(indices, current_depth)

            # left_indices = self.get_indices(tree.left_branch, current_depth + 1)
            # left_indices = left_indices[left_indices != remove_ndx]
            # right_indices = self.get_indices(tree.right_branch, current_depth + 1)
            # right_indices = right_indices[right_indices != remove_ndx]
            # print(left_indices, right_indices, best_attr_ndx)
            # left_branch = self._build_tree(left_indices, current_depth + 1)
            # right_branch = self._build_tree(right_indices, current_depth + 1)
            # self.deletion_type_ = '3'
            # return DecisionNode(feature_i=best_attr_ndx, node_dict=tree.node_dict,
            #                     left_branch=left_branch, right_branch=right_branch)

    def _update_leaves(self, tree, remove_ndx):

        # a leaf! update indices
        if tree.value is not None:
            # print('\nbefore: {}'.format(tree.node_dict['indices']))
            indices = tree.node_dict['indices']
            for i in range(len(indices)):
                if remove_ndx < indices[i]:
                    indices[i] -= 1
            # print('after: {}'.format(tree.node_dict['indices']))

        # keep traversing the tree
        else:
            self._update_leaves(tree.left_branch, remove_ndx)
            self._update_leaves(tree.right_branch, remove_ndx)

    def _update_leaf_node(self, tree, remove_ndx):
        """
        Update this leaf node to effectively remove the target index.
        """
        node_dict = tree.node_dict
        node_dict['pos_count'] -= self.y_train_[remove_ndx]
        node_dict['count'] -= 1
        node_dict['leaf_value'] = 0 if node_dict['pos_count'] == 0 else node_dict['pos_count'] / node_dict['count']
        node_dict['indices'] = node_dict['indices'][node_dict['indices'] != remove_ndx]
        tree.value = node_dict['leaf_value']

    def _update_decision_node(self, node_dict, attr_ndx, abranch, y_val):
        """
        Update the attribute dictionary of the node metadata.
        """

        # access the attriubute metadata
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

        return 1

    # def _update_decision_node(self, branch_dict, y_val, n_samples):
    #     """
    #     Update the attribute split branch metadata.
    #     """

    #     # only the affected instance is in this branch
    #     if branch_dict['count'] <= 1:
    #         return None
    #     else:
    #         branch_dict['count'] -= 1
    #         branch_dict['pos_count'] -= y_val
    #         branch_dict['weight'] = branch_dict['count'] / n_samples
    #         branch_dict['pos_prob'] = branch_dict['pos_count'] / branch_dict['count']
    #         branch_dict['index'] = 1 - (np.square(branch_dict['pos_prob']) + np.square(1 - branch_dict['pos_prob']))
    #         branch_dict['weighted_index'] = branch_dict['weight'] * branch_dict['index']
    #     return 1

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
            print(tree.value, tree.node_dict['indices'], self.y_train_[tree.node_dict['indices']])

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