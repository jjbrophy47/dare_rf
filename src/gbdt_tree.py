"""
Binary GBDT implementation using CART regression trees. MSE is used to find the best split attribute and value
as this is equivalent to finding the maximum variance reduction.
Adapted from MLFromScratch: https://github.com/eriklindernoren/ML-From-Scratch
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


class Tree(object):
    """
    Regression tree specifically for GBDT.

    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    """
    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=4):
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.max_error = 1e6

    def __str__(self):
        s = 'GBDT Tree:'
        s += '\nmin_samples_split={}'.format(self.min_samples_split)
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

        max_error = len(indices)
        smallest_error = max_error
        n_samples = len(indices)
        ndx_map = indices.copy()

        # additional data structure to maintain attribute split info
        node_dict = {}

        # error checking
        if n_samples >= self.min_samples_split and current_depth < self.max_depth:

            # iterate through each feature, sorted by value
            for i in range(self.n_features_):
                node_dict[i] = {}
                sort_ndx = np.argsort(self.X_train_[indices][:, i])
                sorted_values = self.X_train_[indices][:, i][sort_ndx]
                unique_values = np.unique(sorted_values)

                # iterate through unique values of feature i and calculate the error
                for threshold in unique_values:
                    node_dict[i][threshold] = {'left': {'y_sum': 0, 'y_count': 0, 'indices': [], 'ssle': 0.0},
                                               'right': {'y_sum': 0, 'y_count': 0, 'indices': [], 'ssle': 0.0}}
                    split_ndx = self._divide_on_feature(sorted_values, threshold)
                    left_indices = ndx_map[sort_ndx[:split_ndx]]
                    right_indices = ndx_map[sort_ndx[split_ndx:]]

                    # make sure there is atleast 1 sample in each branch
                    if len(left_indices) > 0 and len(right_indices) > 0:
                        y1 = self.y_train_[left_indices]
                        y2 = self.y_train_[right_indices]

                        # calculate sum square loss error
                        computations = self._sum_square_loss_error(y1, y2)
                        y1_sum, y1_count, y1_ssle, y2_sum, y2_count, y2_ssle, error = computations

                        # print(i, threshold, y1_sum, y1_count, y1_ssle)

                        # save attribute split metadata
                        # node_dict['y_count'] = len(left_indices) + len(right_indices)
                        node_dict[i][threshold]['left']['y_sum'] = y1_sum
                        node_dict[i][threshold]['left']['y_count'] = y1_count
                        node_dict[i][threshold]['left']['indices'] = left_indices
                        node_dict[i][threshold]['left']['ssle'] = y1_ssle

                        node_dict[i][threshold]['right']['y_sum'] = y2_sum
                        node_dict[i][threshold]['right']['y_count'] = y2_count
                        node_dict[i][threshold]['right']['indices'] = right_indices
                        node_dict[i][threshold]['right']['ssle'] = y2_ssle

                        # save the attribute, split with the lowest error
                        if error < smallest_error:
                            smallest_error = error
                            best_feature_ndx = i
                            best_feature_threshold = threshold
                            best_left_indices = left_indices
                            best_right_indices = right_indices

                    # print()
                    # print(i, threshold)
                    # print(node_dict[i][threshold]['left'])
                    # print(node_dict[i][threshold]['right'])

        if smallest_error < max_error:
            left_node = self._build_tree(best_left_indices, current_depth + 1)
            right_node = self._build_tree(best_right_indices, current_depth + 1)
            return DecisionNode(feature_i=best_feature_ndx, threshold=best_feature_threshold,
                                left_branch=left_node, right_branch=right_node, node_dict=node_dict)

        # we're at a leaf => determine value
        leaf_value = np.mean(self.y_train_[indices])
        node_dict['y_sum'] = np.sum(self.y_train_[indices])
        node_dict['y_count'] = len(indices)
        return DecisionNode(value=leaf_value, node_dict=node_dict)


    def _divide_on_feature(self, sorted_values, threshold):
        """
        Return split index based on the threshold.
        """
        split_ndx = len(np.where(sorted_values <= threshold)[0])
        return split_ndx


    def _sum_square_loss_error(self, y1, y2):
        """
        From paper: On Incremental Learning for Gradient Boosting Decision Trees, formula (3).
        """
        y1_sum = np.sum(y1)
        y1_count = len(y1)
        y1_ssle = np.sum(np.square(y1 - (y1_sum / y1_count)))

        y2_sum = np.sum(y2)
        y2_count = len(y2)
        y2_ssle = np.sum(np.square(y2 - (y2_sum / y2_count)))

        objective = y1_ssle + y2_ssle
        return y1_sum, y1_count, y1_ssle, y2_sum, y2_count, y2_ssle, objective

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
        self._delete(x, remove_ndx)

    # TODO: if decision node only contains 2 instances, then remove leaf with the instance to be removed
    # to prevent rebuilding the subtree.
    def _delete(self, x, remove_ndx, tree=None, current_depth=0):

        if tree is None:
            tree = self.root_

        # update leaf metadata
        elif tree.value is not None:
            self._update_leaf_node(tree, remove_ndx)
            return

        node_dict = tree.node_dict
        original_attr_ndx = tree.feature_i
        original_attr_threshold = tree.threshold
        original_objective = node_dict[tree.feature_i][tree.threshold]['left']['ssle'] + \
                             node_dict[tree.feature_i][tree.threshold]['right']['ssle']
        smallest_error = X.shape[0]

        best_attr_ndx = None
        best_attr_threshold = None

        print('\n\nCurrent attribute, threshold: {}, {}'.format(original_attr_ndx, original_attr_threshold))

        # update the error for each attribute split
        for attr_ndx in node_dict.keys():
            for attr_threshold in node_dict[attr_ndx].keys():
                print('\nchecking feature: {}, threshold: {}'.format(attr_ndx, attr_threshold))

                if node_dict[attr_ndx][attr_threshold]['left']['y_count'] == 0 or \
                   node_dict[attr_ndx][attr_threshold]['right']['y_count'] == 0:
                   print('invalid split')
                   continue

                # only update the affected branch
                branch = 'left' if x[attr_ndx] <= attr_threshold else 'right'
                branch_dict = node_dict[attr_ndx][attr_threshold][branch]

                print(node_dict[attr_ndx][attr_threshold]['left'])
                print(node_dict[attr_ndx][attr_threshold]['right'])

                # affected branch contains less than the min split requirements
                if branch_dict['y_count'] < 2:
                        print('less than 2')
                        continue
                else:
                    one_valid_split = True
                self._update_decision_node(branch_dict, remove_ndx)
                print(node_dict[attr_ndx][attr_threshold][branch])

                # recompute error for this attribute split
                split_dict = node_dict[attr_ndx][attr_threshold]
                split_error = split_dict['left']['ssle'] + split_dict['right']['ssle']
                print('new_error: {}'.format(split_error))

                # save this attribute split if it is the best
                if split_error < smallest_error:
                    smallest_error = split_error
                    best_attr_ndx = attr_ndx
                    best_attr_threshold = attr_threshold
                    affected_branch = branch

        # if there are no valid splits, remove leaf with the instance to be removed,
        # then absorb the other leaf into the parent node.
        # this will likely only happen when there are 2 instances in the node,
        # one for each leaf.

        # keep original attribute split if it is tied for the smallest error after removal
        if node_dict[tree.feature_i][tree.threshold]['left']['ssle'] +\
           node_dict[tree.feature_i][tree.threshold]['right']['ssle'] == smallest_error:
           best_attr_ndx = tree.feature_i
           best_attr_threshold = tree.threshold
           affected_branch = 'left' if x[best_attr_ndx] <= best_attr_threshold else 'right'

        print('\nOld attribute, threshold: {}, {}'.format(original_attr_ndx, original_attr_threshold))
        print('New attribute, threshold: {}, {}'.format(best_attr_ndx, best_attr_threshold))

        # check to see if the tree needs to be restructured
        print(best_attr_ndx, original_attr_ndx)
        if best_attr_ndx == original_attr_ndx:

            # check is done for this node, traverse affected branch and the check the remaining nodes
            # TODO: no need to rebuild if same attribute and threshold is one neighbor value below original
            print(best_attr_threshold, original_attr_threshold)
            if best_attr_threshold == original_attr_threshold:
                print('check complete at depth {}, traversing {}'.format(current_depth, affected_branch))
                tree_branch = tree.left_branch if affected_branch == 'left' else tree.right_branch

                # print(best_attr_ndx, best_attr_threshold)
                # print(node_dict)

                # turn decision node into a leaf if the affected branch only contains the instance to remove
                if node_dict[tree.feature_i][tree.threshold][affected_branch]['y_count'] == 1:
                    na_branch = 'left' if affected_branch == 'right' else 'right'
                    branch_indices = node_dict[tree.feature_i][tree.threshold][na_branch]['indices']
                    branch_dict = {'y_sum': self.y_train_[branch_indices], 'y_count': 1}
                    branch_dict['leaf_value'] = branch_dict['y_sum']
                    tree_branch = DecisionNode(value=branch_dict['leaf_value'], node_dict=branch_dict)


                # # turn affected branch into a leaf if it only contains 2 instances
                # if node_dict[tree.feature_i][tree.threshold]['left']['y_count'] + \
                #    node_dict[tree.feature_i][tree.threshold]['right']['y_count'] == 2:
                #     na_branch = 'left' if affected_branch == 'right' else 'right'
                #     branch_indices = node_dict[tree.feature_i][tree.threshold][na_branch]['indices']
                #     branch_dict = {'y_sum': self.y_train_[branch_indices], 'y_count': 1}
                #     branch_dict['leaf_value'] = branch_dict['y_sum']
                #     tree_branch = DecisionNode(value=branch_dict['leaf_value'], node_dict=branch_dict)

                # traverse the affected branch and continue checking
                else:
                    self._delete(x, remove_ndx, tree=tree_branch, current_depth=current_depth + 1)

            # split data left and right, rebuild subtree below this node
            else:
                print('rebuilding at depth {}'.format(current_depth))
                # print(best_attr_ndx, best_attr_threshold)
                branch_dict = node_dict[best_attr_ndx][best_attr_threshold]
                left_node = self._build_tree(branch_dict['left']['indices'], current_depth + 1)
                right_node = self._build_tree(branch_dict['right']['indices'], current_depth + 1)
                tree = DecisionNode(feature_i=best_attr_ndx, threshold=best_attr_threshold,
                                    left_branch=left_node, right_branch=right_node, node_dict=node_dict)

        # rebuild subtree below this node
        else:
            print('rebuilding at depth {}'.format(current_depth))
            # print(best_attr_ndx, best_attr_threshold)
            branch_dict = node_dict[best_attr_ndx][best_attr_threshold]
            left_node = self._build_tree(branch_dict['left']['indices'], current_depth + 1)
            right_node = self._build_tree(branch_dict['right']['indices'], current_depth + 1)
            tree = DecisionNode(feature_i=best_attr_ndx, threshold=best_attr_threshold,
                                left_branch=left_node, right_branch=right_node, node_dict=node_dict)

    def _update_leaf_node(self, tree, remove_ndx):
        node_dict = tree.node_dict
        node_dict['y_sum'] -= self.y_train_[remove_ndx]
        node_dict['y_count'] -= 1
        node_dict['leaf_value'] = 0 if node_dict['y_sum'] == 0 else node_dict['y_sum'] / node_dict['y_count']
        tree.value = node_dict['leaf_value']

    def _update_decision_node(self, branch_dict, remove_ndx):
        """
        Update the attribute split branch metadata.
        """
        branch_dict['y_sum'] -= self.y_train_[remove_ndx]
        branch_dict['y_count'] -= 1
        branch_dict['indices'] = np.delete(branch_dict['indices'], np.where(branch_dict['indices'] == remove_ndx))
        branch_dict['ssle'] = np.sum(np.square(self.y_train_[branch_dict['indices']] - \
                                    (branch_dict['y_sum'] / branch_dict['y_count'])))


    def print_tree(self, tree=None, indent='\t', depth=0):
        """
        Recursively print the decision tree.
        """
        if tree is None:
            tree = self.root_

        indent_str = indent * depth

        # If we're at leaf => print the label
        if tree.value is not None:
            print(tree.value)

        # Go deeper down the tree
        else:

            # Print test
            print("%s:%s? " % (tree.feature_i, tree.threshold))

            # Print the left branch
            print("%sT->" % (indent_str), end="")
            self.print_tree(tree.left_branch, depth=depth + 1)

            # Print the right branch
            print("%sF->" % (indent_str), end="")
            self.print_tree(tree.right_branch, depth=depth + 1)

    def equals(self, this=None, other=None):
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
            return 1 if this.feature_i == other.feature_i and this.threshold == other.threshold \
                   and self.equals(this.left_branch, other.left_branch) \
                   and self.equals(this.right_branch, other.right_branch) else 0


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

        # choose the feature that we will test
        feature_value = x[tree.feature_i]

        assert isinstance(feature_value, int) or isinstance(feature_value, float)
        if feature_value <= tree.threshold:
            branch = tree.left_branch

        elif feature_value > tree.threshold:
            branch = tree.right_branch

        # test subtree
        return self._predict_value(x, branch)


if __name__ == '__main__':
    n_samples = 10
    # X = np.array([[ 0.98460299,  0.53113189],
    #      [-0.5863483,   0.61571276],
    #      [-1.0335856,   0.31619801],
    #      [ 0.19937955,  0.7161298 ],
    #      [-0.17281627,  1.31424692],
    #      [-1.4520193,   0.27434247],
    #      [ 2.25515576, -0.86252171],
    #      [-2.80359674,  0.190064  ],
    #      [ 0.57356042, -1.7271028 ],
    #      [ 0.20179,    -0.22429964]])
    # y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0])
 #    X = np.array([[ 2.11080783, -1.55389423],
 # [ 0.70891831,  0.64160989],
 # [ 0.39329549, -1.23133961],
 # [-0.75385227, -1.083754  ],
 # [-0.87258163,  0.77069556],
 # [-0.44259817,  0.93891464],
 # [ 0.82132568, -0.21707077],
 # [ 1.12852667, -0.55227303],
 # [-0.07470203,  0.69973125],
 # [-0.54179079, -0.5844761 ]])
    # y = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])
    X = np.random.randn(n_samples, 2)
    y = np.random.randint(2, size=n_samples)
    t = Tree(max_depth=4).fit(X, y)
    t.print_tree()
    print(X, y)
    print(t.predict(X[[0]]), y[0])

    print(X[0])
    t.delete(0)
