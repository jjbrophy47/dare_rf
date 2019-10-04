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
    def __init__(self, feature_i=None, threshold=None, value=None, left_branch=None, right_branch=None):
        self.feature_i = feature_i          # Index for the feature that is tested
        self.threshold = threshold          # Threshold value for feature
        self.value = value                  # Value if the node is a leaf in the tree
        self.left_branch = left_branch      # 'Left' subtree
        self.right_branch = right_branch    # 'Right' subtree


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

        # error checking
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:

            attr_val_dict = {}

            # sort each feature by value
            for i in range(self.n_features_):
                sort_ndx = np.argsort(self.X_train_[indices][:, i])
                sorted_values = self.X_train_[indices][:, i][sort_ndx]
                unique_values = np.unique(sorted_values)

                # iterate through unique values of feature i and calculate the error
                for threshold in unique_values:
                    split_ndx = self._divide_on_feature(sorted_values, threshold)
                    left_indices = ndx_map[sort_ndx[:split_ndx]]
                    right_indices = ndx_map[sort_ndx[split_ndx:]]

                    # make sure there is atleast 1 sample in each branch
                    if len(left_indices) > 0 and len(right_indices) > 0:
                        y1 = self.y_train_[left_indices]
                        y2 = self.y_train_[right_indices]

                        # calculate sum square loss error
                        error = self._sum_square_loss_error(y1, y2)

                        # save the attribute, split with the lowest error
                        if error < smallest_error:
                            smallest_error = error
                            best_feature_ndx = i
                            best_feature_threshold = threshold
                            best_left_indices = left_indices
                            best_right_indices = right_indices

        if smallest_error < max_error:
            left_node = self._build_tree(best_left_indices, current_depth + 1)
            right_node = self._build_tree(best_right_indices, current_depth + 1)
            return DecisionNode(feature_i=best_feature_ndx, threshold=best_feature_threshold,
                                left_branch=left_node, right_branch=right_node)

        # we're at a leaf => determine value
        leaf_value = np.mean(self.y_train_[indices])
        return DecisionNode(value=leaf_value)


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
        s1 = np.sum(np.square(y1 - np.mean(y1)))
        s2 = np.sum(np.square(y2 - np.mean(y2)))
        return s1 + s2

    def predict(self, X):
        """
        Classify samples one by one and return the set of labels.
        """
        assert X.ndim == 2
        y_pred = [self._predict_value(sample) for sample in X]
        return y_pred

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



# class XGBoostRegressionTree(DecisionTree):
#     """
#     Regression tree for XGBoost
#     - Reference -
#     http://xgboost.readthedocs.io/en/latest/model.html
#     """

#     def _split(self, y):
#         """ y contains y_true in left half of the middle column and
#         y_pred in the right half. Split and return the two matrices """
#         col = int(np.shape(y)[1]/2)
#         y, y_pred = y[:, :col], y[:, col:]
#         return y, y_pred

#     def _gain(self, y, y_pred):
#         nominator = np.power((y * self.loss.gradient(y, y_pred)).sum(), 2)
#         denominator = self.loss.hess(y, y_pred).sum()
#         return 0.5 * (nominator / denominator)

#     def _gain_by_taylor(self, y, y1, y2):
#         # Split
#         y, y_pred = self._split(y)
#         y1, y1_pred = self._split(y1)
#         y2, y2_pred = self._split(y2)

#         true_gain = self._gain(y1, y1_pred)
#         false_gain = self._gain(y2, y2_pred)
#         gain = self._gain(y, y_pred)
#         return true_gain + false_gain - gain

#     def _approximate_update(self, y):
#         # y split into y, y_pred
#         y, y_pred = self._split(y)
#         # Newton's Method
#         gradient = np.sum(y * self.loss.gradient(y, y_pred), axis=0)
#         hessian = np.sum(self.loss.hess(y, y_pred), axis=0)
#         update_approximation =  gradient / hessian

#         return update_approximation

#     def fit(self, X, y):
#         self._impurity_calculation = self._gain_by_taylor
#         self._leaf_value_calculation = self._approximate_update
#         super(XGBoostRegressionTree, self).fit(X, y)


# class RegressionTree(DecisionTree):
#     def _calculate_variance_reduction(self, y, y1, y2):
#         var_tot = calculate_variance(y)
#         var_1 = calculate_variance(y1)
#         var_2 = calculate_variance(y2)
#         frac_1 = len(y1) / len(y)
#         frac_2 = len(y2) / len(y)

#         # Calculate the variance reduction
#         variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)

#         return sum(variance_reduction)

#     def _mean_of_y(self, y):
#         value = np.mean(y, axis=0)
#         return value if len(value) > 1 else value[0]

#     def fit(self, X, y):
#         self._impurity_calculation = self._calculate_variance_reduction
#         self._leaf_value_calculation = self._mean_of_y
#         super(RegressionTree, self).fit(X, y)

# class ClassificationTree(DecisionTree):
#     def _calculate_information_gain(self, y, y1, y2):
#         # Calculate information gain
#         p = len(y1) / len(y)
#         entropy = calculate_entropy(y)
#         info_gain = entropy - p * \
#             calculate_entropy(y1) - (1 - p) * \
#             calculate_entropy(y2)

#         return info_gain

#     def _majority_vote(self, y):
#         most_common = None
#         max_count = 0
#         for label in np.unique(y):
#             # Count number of occurences of samples with label
#             count = len(y[y == label])
#             if count > max_count:
#                 most_common = label
#                 max_count = count
#         return most_common

#     def fit(self, X, y):
#         self._impurity_calculation = self._calculate_information_gain
#         self._leaf_value_calculation = self._majority_vote
#         super(ClassificationTree, self).fit(X, y)


if __name__ == '__main__':
    X = np.random.randn(10, 2)
    y = np.random.randint(2, size=10)
    t = Tree(max_depth=4).fit(X, y)
    t.print_tree()
    print(t.predict(X[[0]]))
