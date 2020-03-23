"""
CeDAR binary tree implementation; only supports binary attributes and a binary label.
Represenation is a number of parllel arrays.
Adapted from: https://github.com/scikit-learn/scikit-learn/blob/b194674c42d54b26137a456c510c5fdba1ba23e0/sklearn/tree/_tree.pyx
"""
from cpython cimport Py_INCREF
from cpython.ref cimport PyObject

cimport cython

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf
from libc.math cimport exp
from libc.time cimport time

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport compute_gini
from ._utils cimport generate_distribution
from ._utils cimport convert_int_ndarray
from ._utils cimport dealloc

cdef int UNDEF = -1

# =====================================
# Remover
# =====================================

cdef class _Remover:
    """
    Removes data from a learned tree.
    """

    # removal metrics
    property remove_types:
        def __get__(self):
            return self._get_int_ndarray(self.remove_types, self.remove_count)

    property remove_depths:
        def __get__(self):
            return self._get_int_ndarray(self.remove_depths, self.remove_count)

    def __cinit__(self, _DataManager manager, _TreeBuilder tree_builder,
                  double epsilon, double lmbda):
        """
        Constructor.
        """
        self.manager = manager
        self.tree_builder = tree_builder
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.min_samples_leaf = tree_builder.min_samples_leaf
        self.min_samples_split = tree_builder.min_samples_split

        self.capacity = 10
        self.remove_count = 0
        self.remove_types = <int *>malloc(self.capacity * sizeof(int))
        self.remove_depths = <int *>malloc(self.capacity * sizeof(int))

    def __dealloc__(self):
        """
        Destructor.
        """
        if self.remove_types:
            free(self.remove_types)
        if self.remove_depths:
            free(self.remove_depths)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int remove(self, _Tree tree, np.ndarray remove_indices):
        """
        Remove the data from remove_indices from the learned _Tree.
        """

        # Parameters
        cdef _DataManager manager = self.manager

        # get data
        cdef int** X = NULL
        cdef int* y = NULL

        cdef int* samples = convert_int_ndarray(remove_indices)
        cdef int  n_samples = remove_indices.shape[0]
        cdef int  result = 0

        cdef int  remove_count = 0
        cdef int* remove_types = <int *>malloc(n_samples * sizeof(int))
        cdef int* remove_depths = <int *>malloc(n_samples * sizeof(int))

        # make room for new deletions
        self._resize_metrics(n_samples)

        # check if any sample has already been deleted
        result = manager.check_sample_validity(samples, n_samples)
        if result == -1:
            return -1
        manager.get_data(&X, &y)

        self._remove(&tree.root, X, y, samples, n_samples, 1.0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _remove(self, Node** node_ptr, int** X, int* y,
                      int* samples, int n_samples, double parent_p) nogil:
        """
        Recusrively remove the samples from this subtree.
        """

        cdef Node *node = node_ptr[0]

        cdef SplitRecord split
        cdef int result = 0

        cdef int* leaf_samples = NULL
        cdef int  leaf_samples_count

        # printf("popping_r (%d, %d, %.7f, %d, %d, %d)\n", node.depth, node.is_left,
        #        parent_p, node.count, n_samples, node.feature_count)

        # leaf
        if node.is_leaf:
            self._update_leaf(node_ptr, y, samples, n_samples)
            self._add_removal_type(result, node.depth)

        # decision node
        else:

            # compute new statistics
            result = self._node_remove(node, X, y, samples, n_samples, parent_p, &split)
            # printf('result: %d\n', result)

            # convert to leaf
            if result == 1:
                self._convert_to_leaf(node_ptr, samples, n_samples, &split)
                self._add_removal_type(result, node.depth)

            # retrain
            elif result == 2:
                leaf_samples = <int *>malloc(split.count * sizeof(int))
                leaf_samples_count = 0
                self._get_leaf_samples(node, samples, n_samples,
                                       &leaf_samples, &leaf_samples_count)
                dealloc(node)
                node_ptr[0] = self.tree_builder._build(X, y, leaf_samples, leaf_samples_count,
                                                       split.valid_features, split.feature_count,
                                                       node.depth, node.is_left, parent_p)
                self._add_removal_type(result, node.depth)

            # update and recurse
            else:
                self._update_decision_node(node_ptr, &split)

                # traverse left
                if split.left_count > 0:
                    self._remove(&node.left, X, y, split.left_indices,
                                 split.left_count, split.p)

                # traverse right
                if split.right_count > 0:
                    self._remove(&node.right, X, y, split.right_indices,
                                 split.right_count, split.p)

        free(samples)

    # private
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _update_leaf(self, Node** node_ptr, int* y, int* samples, int n_samples) nogil:
        """
        Update leaf node: count, pos_count, value, leaf_samples.
        """
        cdef Node* node = node_ptr[0]
        cdef int pos_count = 0

        # count number of pos labels in removal data
        for i in range(n_samples):
            if y[samples[i]] == 1:
                pos_count += 1

        # remove samples from leaf
        cdef int* leaf_samples = <int *>malloc((node.count - n_samples) * sizeof(int))
        cdef int  leaf_samples_count = 0
        self._get_leaf_samples(node, samples, n_samples, &leaf_samples, &leaf_samples_count)
        free(node.leaf_samples)

        node.count -= n_samples
        node.pos_count -= pos_count
        node.value = node.pos_count / <double> node.count
        node.leaf_samples = leaf_samples

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _convert_to_leaf(self, Node** node_ptr, int* samples, int n_samples,
                               SplitRecord *split) nogil:
        """
        Convert decision node to a leaf node.
        """
        cdef Node* node = node_ptr[0]

        cdef int* leaf_samples = <int *>malloc(split.count * sizeof(int))
        cdef int  leaf_samples_count = 0
        self._get_leaf_samples(node, samples, n_samples, &leaf_samples, &leaf_samples_count)
        dealloc(node)

        node.count = split.count
        node.pos_count = split.pos_count

        node.is_leaf = 1
        node.value = node.pos_count / <double> node.count
        node.leaf_samples = leaf_samples

        node.p = UNDEF
        node.feature = UNDEF
        node.feature_count = UNDEF
        node.valid_features = NULL
        node.left_counts = NULL
        node.left_pos_counts = NULL
        node.right_counts = NULL
        node.right_pos_counts = NULL

        node.left = NULL
        node.right = NULL

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _get_leaf_samples(self, Node* node, int* remove_samples, int n_remove_samples,
                                int** leaf_samples_ptr, int* leaf_samples_count_ptr) nogil:
        """
        Recursively obtain and filter the samples at the leaves.
        """
        cdef bint add_sample

        if node.is_leaf:
            for i in range(node.count):
                add_sample = 1

                for j in range(n_remove_samples):
                    if node.leaf_samples[i] == remove_samples[j]:
                        add_sample = 0
                        break

                if add_sample:
                    leaf_samples_ptr[0][leaf_samples_count_ptr[0]] = node.leaf_samples[i]
                    leaf_samples_count_ptr[0] += 1

        else:
            if node.left:
                self._get_leaf_samples(node.left, remove_samples, n_remove_samples,
                                       leaf_samples_ptr, leaf_samples_count_ptr)

            if node.right:
                self._get_leaf_samples(node.right, remove_samples, n_remove_samples,
                                       leaf_samples_ptr, leaf_samples_count_ptr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _update_decision_node(self, Node** node_ptr, SplitRecord *split) nogil:
        """
        Update tree with node metadata.
        """
        cdef Node* node = node_ptr[0]
        node.p = split.p
        node.count = split.count
        node.pos_count = split.pos_count
        node.feature_count = split.feature_count
        node.valid_features = split.valid_features
        node.left_counts = split.left_counts
        node.left_pos_counts = split.left_pos_counts
        node.right_counts = split.right_counts
        node.right_pos_counts = split.right_pos_counts


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int _node_remove(self, Node* node, int** X, int* y,
                          int* samples, int n_samples,
                          double parent_p, SplitRecord *split) nogil:
        """
        Update node statistics based on the removal data (X, y).
        Return 0 for a successful update, 1 to signal a leaf creation,
          2 to signal a retrain.
        """

        # parameters
        cdef double epsilon = self.epsilon
        cdef double lmbda = self.lmbda
        cdef int min_samples_leaf = self.min_samples_leaf
        cdef int min_samples_split = self.min_samples_split

        # removal data statistics
        cdef int count = n_samples
        cdef int pos_count = 0
        cdef int left_count
        cdef int left_pos_count
        cdef int right_count
        cdef int right_pos_count

        # overall data statistics
        cdef int updated_count
        cdef int updated_pos_count
        cdef int updated_left_count
        cdef int updated_left_pos_count
        cdef int updated_right_count
        cdef int updated_right_pos_count

        cdef int i
        cdef int j
        cdef int k

        cdef int feature_count = 0
        cdef int result = 0

        cdef bint chosen_feature_validated = 0
        cdef int chosen_ndx

        cdef double p
        cdef double ratio

        cdef double* gini_indices
        cdef double* distribution
        cdef int* valid_features

        cdef int* left_counts
        cdef int* left_pos_counts
        cdef int* right_counts
        cdef int* right_pos_counts

        cdef int chosen_left_count
        cdef int chosen_right_count

        # count number of pos labels in removal data
        for i in range(n_samples):
            if y[samples[i]] == 1:
                pos_count += 1

        updated_count = node.count - count
        updated_pos_count = node.pos_count - pos_count

        # no samples left in this node => retrain
        if updated_count <= 0:  # this branch will not be reached
            # printf('meta count <= count\n')
            result = 2

        # only samples from one class are left in this node => create leaf
        elif updated_pos_count == 0 or updated_pos_count == updated_count:
            # printf('only samples from one class\n')
            result = 1
            split.count = updated_count
            split.pos_count = updated_pos_count

        else:

            gini_indices = <double *>malloc(node.feature_count * sizeof(double))
            distribution = <double *>malloc(node.feature_count * sizeof(double))
            valid_features = <int *>malloc(node.feature_count * sizeof(int))

            updated_left_counts = <int *>malloc(node.feature_count * sizeof(int))
            updated_left_pos_counts = <int *>malloc(node.feature_count * sizeof(int))
            updated_right_counts = <int *>malloc(node.feature_count * sizeof(int))
            updated_right_pos_counts = <int *>malloc(node.feature_count * sizeof(int))

            # compute statistics of the removal data for each attribute
            for j in range(node.feature_count):

                left_count = 0
                left_pos_count = 0

                for i in range(n_samples):

                    if X[samples[i]][node.valid_features[j]] == 1:
                        left_count += 1
                        left_pos_count += y[samples[i]]

                right_count = count - left_count
                right_pos_count = pos_count - left_pos_count

                updated_left_count = node.left_counts[j] - left_count
                updated_left_pos_count = node.left_pos_counts[j] - left_pos_count
                updated_right_count = node.right_counts[j] - right_count
                updated_right_pos_count = node.right_pos_counts[j] - right_pos_count

                # validate split
                if updated_left_count >= min_samples_leaf and updated_right_count >= min_samples_leaf:
                    valid_features[feature_count] = node.valid_features[j]
                    gini_indices[feature_count] = compute_gini(updated_count, updated_left_count,
                        updated_right_count, updated_left_pos_count, updated_right_pos_count)

                    # update metadata
                    updated_left_counts[feature_count] = updated_left_count
                    updated_left_pos_counts[feature_count] = updated_left_pos_count
                    updated_right_counts[feature_count] = updated_right_count
                    updated_right_pos_counts[feature_count] = updated_right_pos_count

                    if node.valid_features[j] == node.feature:
                        chosen_feature_validated = 1
                        chosen_ndx = feature_count
                        chosen_left_count = left_count
                        chosen_right_count = right_count

                    feature_count += 1

            # no valid features after data removal => create leaf
            if feature_count == 0:
                # printf('feature_count is zero\n')
                result = 1
                free(gini_indices)
                free(distribution)
                free(valid_features)

                free(updated_left_counts)
                free(updated_left_pos_counts)
                free(updated_right_counts)
                free(updated_right_pos_counts)

                split.count = updated_count
                split.pos_count = updated_pos_count

            # current feature no longer valid => retrain
            elif not chosen_feature_validated:
                # printf('chosen feature not validated\n')
                result = 2
                free(gini_indices)
                free(distribution)

                free(updated_left_counts)
                free(updated_left_pos_counts)
                free(updated_right_counts)
                free(updated_right_pos_counts)

                split.count = updated_count
                split.feature_count = feature_count
                split.valid_features = valid_features

            else:

                # remove invalid features
                gini_indices = <double *>realloc(gini_indices, feature_count * sizeof(double))
                distribution = <double *>realloc(distribution, feature_count * sizeof(double))
                valid_features = <int *>realloc(valid_features, feature_count * sizeof(int))

                updated_left_counts = <int *>realloc(updated_left_counts, feature_count * sizeof(int))
                updated_left_pos_counts = <int *>realloc(updated_left_pos_counts, feature_count * sizeof(int))
                updated_right_counts = <int *>realloc(updated_right_counts, feature_count * sizeof(int))
                updated_right_pos_counts = <int *>realloc(updated_right_pos_counts, feature_count * sizeof(int))

                # compute new probability for the chosen feature
                generate_distribution(lmbda, distribution, gini_indices, feature_count)
                p = parent_p * distribution[chosen_ndx]
                ratio = p / node.p

                # printf('ratio: %.3f, epsilon: %.3f, lmbda: %.3f\n', ratio, epsilon, lmbda)

                # compare with previous probability => retrain if necessary
                if ratio < exp(-epsilon) or ratio > exp(epsilon):
                    # printf('bounds exceeded\n')
                    result = 2
                    free(gini_indices)
                    free(distribution)

                    free(updated_left_counts)
                    free(updated_left_pos_counts)
                    free(updated_right_counts)
                    free(updated_right_pos_counts)

                    split.count = updated_count
                    split.feature_count = feature_count
                    split.valid_features = valid_features

                else:

                    # split removal data based on the chosen feature
                    split.left_indices = <int *>malloc(chosen_left_count * sizeof(int))
                    split.right_indices = <int *>malloc(chosen_right_count * sizeof(int))
                    j = 0
                    k = 0
                    for i in range(n_samples):
                        if X[samples[i]][valid_features[chosen_ndx]] == 1:
                            split.left_indices[j] = samples[i]
                            j += 1
                        else:
                            split.right_indices[k] = samples[i]
                            k += 1
                    split.left_count = j
                    split.right_count = k

                    # cleanup
                    free(gini_indices)
                    free(distribution)

                    split.p = p
                    split.count = updated_count
                    split.pos_count = updated_pos_count
                    split.feature_count = feature_count
                    split.valid_features = valid_features
                    split.left_counts = updated_left_counts
                    split.left_pos_counts = updated_left_pos_counts
                    split.right_counts = updated_right_counts
                    split.right_pos_counts = updated_right_pos_counts

        return result

    cdef void _resize_metrics(self, int capacity=0) nogil:
        """
        Increase size of removal allocations.
        """
        if capacity > self.capacity - self.remove_count:
            if self.capacity * 2 - self.remove_count > capacity:
                self.capacity *= 2
            else:
                self.capacity = int(capacity)

        # removal info
        if self.remove_types and self.remove_depths:
            self.remove_types = <int *>realloc(self.remove_types, self.capacity * sizeof(int))
            self.remove_depths = <int *>realloc(self.remove_depths, self.capacity * sizeof(int))
        else:
            self.remove_types = <int *>malloc(self.capacity * sizeof(int))
            self.remove_depths = <int *>malloc(self.capacity * sizeof(int))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _add_removal_type(self, int remove_type, int remove_depth) nogil:
        """
        Adds to the removal metrics.
        """
        self.remove_types[self.remove_count] = remove_type
        self.remove_depths[self.remove_count] = remove_depth
        self.remove_count += 1

    cpdef void clear_removal_metrics(self):
        """
        Resets deletion statistics.
        """
        free(self.remove_types)
        free(self.remove_depths)
        self.remove_count = 0
        self.remove_types = NULL
        self.remove_depths = NULL

    cdef np.ndarray _get_int_ndarray(self, int *data, int n_elem):
        """
        Wraps value as a 1-d NumPy array.
        The array keeps a reference to this Tree, which manages the underlying memory.
        """
        cdef np.npy_intp shape[1]
        shape[0] = n_elem
        cdef np.ndarray arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, data)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr
