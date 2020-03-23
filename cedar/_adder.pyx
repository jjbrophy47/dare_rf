"""
Module that adds data to a CeDAR tree.
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

cdef class _Adder:
    """
    Adds data to a learned tree.
    """

    # add metrics
    property add_types:
        def __get__(self):
            return self._get_int_ndarray(self.add_types, self.add_count)

    property add_depths:
        def __get__(self):
            return self._get_int_ndarray(self.add_depths, self.add_count)

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
        self.max_depth = tree_builder.max_depth

        self.capacity = 10
        self.add_count = 0
        self.add_types = <int *>malloc(self.capacity * sizeof(int))
        self.add_depths = <int *>malloc(self.capacity * sizeof(int))

    def __dealloc__(self):
        """
        Destructor.
        """
        if self.add_types:
            free(self.add_types)
        if self.add_depths:
            free(self.add_depths)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int add(self, _Tree tree):
        """
        Add the data to the learned _Tree.
        """

        # Parameters
        cdef int* samples = self.manager.get_add_indices()
        cdef int  n_samples = self.manager.n_add_indices

        # Data containers
        cdef int** X = NULL
        cdef int* y = NULL

        self.manager.get_data(&X, &y)
        self._resize_metrics(n_samples)
        self._add(&tree.root, X, y, samples, n_samples, 1.0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _add(self, Node** node_ptr, int** X, int* y,
                   int* samples, int n_samples, double parent_p) nogil:
        """
        Recusrively remove the samples from this subtree.
        """

        cdef Node *node = node_ptr[0]

        cdef SplitRecord split
        cdef int result = 0

        cdef int* leaf_samples = NULL
        cdef int  leaf_samples_count

        # printf("popping_a (%d, %d, %.7f, %d, %d, %d)\n", node.depth, node.is_left,
        #        parent_p, node.count, n_samples, node.feature_count)

        # leaf at max_depth - no expansion possible
        if node.depth == self.max_depth and node.is_leaf:
            self._update_leaf(node_ptr, y, samples, n_samples)
            self._add_removal_type(result, node.depth)

        # decision node or non-max_depth leaf
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

        # count number of pos labels in add data
        for i in range(n_samples):
            if y[samples[i]] == 1:
                pos_count += 1

        # add samples to leaf
        cdef int* leaf_samples = <int *>malloc((node.count - n_samples) * sizeof(int))
        cdef int  leaf_samples_count = 0
        self._get_leaf_samples(node, &leaf_samples, &leaf_samples_count)
        self._add_leaf_samples(samples, n_samples, &leaf_samples, &leaf_samples_count)
        free(node.leaf_samples)

        node.count += n_samples
        node.pos_count += pos_count
        node.value = node.pos_count / <double> node.count
        node.leaf_samples = leaf_samples

    # TODO: not needed; maybe convert_to_decision?
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
    cdef void _get_leaf_samples(self, Node* node, int** leaf_samples_ptr,
                                int* leaf_samples_count_ptr) nogil:
        """
        Recursively obtain all leaf samples from this subtree.
        """

        if node.is_leaf:
            for i in range(node.count):
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
    cdef void _add_leaf_samples(self, int* samples, int n_samples,
                                int** leaf_samples_ptr,
                                int*  leaf_samples_count_ptr) nogil:
        """
        Add samples to leaf samples.
        """
        cdef int i

        for i in range(n_samples):
            leaf_samples_ptr[0][leaf_samples_count_ptr[0]] = samples[i]
            leaf_samples_count_ptr[0] += 1

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
    cdef int _node_add(self, Node* node, int** X, int* y,
                       int* samples, int n_samples,
                       double parent_p, SplitRecord *split) nogil:
        """
        Update node statistics based on the removal data (X, y).
        Return 0 for a successful update,
               1 to signal a decision node creation,
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

        # count number of pos labels in add data
        for i in range(n_samples):
            if y[samples[i]] == 1:
                pos_count += 1

        updated_count = node.count + count
        updated_pos_count = node.pos_count + pos_count

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

        # leaf still has no valid features => update leaf
        if feature_count == 0 and node.is_leaf:
            printf('feature_count is zero\n')
            result = 0
            free(gini_indices)
            free(distribution)
            free(valid_features)

            free(updated_left_counts)
            free(updated_left_pos_counts)
            free(updated_right_counts)
            free(updated_right_pos_counts)

            split.count = updated_count
            split.pos_count = updated_pos_count

        # leaf now has valid features => retrain
        elif feature_count > 0 and node.is_leaf:
            printf('leaf has valid features\n')
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

            # # remove invalid features
            # gini_indices = <double *>realloc(gini_indices, feature_count * sizeof(double))
            # distribution = <double *>realloc(distribution, feature_count * sizeof(double))
            # valid_features = <int *>realloc(valid_features, feature_count * sizeof(int))

            # updated_left_counts = <int *>realloc(updated_left_counts, feature_count * sizeof(int))
            # updated_left_pos_counts = <int *>realloc(updated_left_pos_counts, feature_count * sizeof(int))
            # updated_right_counts = <int *>realloc(updated_right_counts, feature_count * sizeof(int))
            # updated_right_pos_counts = <int *>realloc(updated_right_pos_counts, feature_count * sizeof(int))

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
        Increase size of add allocations.
        """
        if capacity > self.capacity - self.add_count:
            if self.capacity * 2 - self.add_count > capacity:
                self.capacity *= 2
            else:
                self.capacity = int(capacity)

        # add info
        if self.add_types and self.add_depths:
            self.add_types = <int *>realloc(self.add_types, self.capacity * sizeof(int))
            self.add_depths = <int *>realloc(self.add_depths, self.capacity * sizeof(int))
        else:
            self.add_types = <int *>malloc(self.capacity * sizeof(int))
            self.add_depths = <int *>malloc(self.capacity * sizeof(int))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _add_add_type(self, int add_type, int add_depth) nogil:
        """
        Adds to the removal metrics.
        """
        self.add_types[self.add_count] = add_type
        self.add_depths[self.add_count] = add_depth
        self.add_count += 1

    cpdef void clear_add_metrics(self):
        """
        Resets addition statistics.
        """
        free(self.add_types)
        free(self.add_depths)
        self.add_count = 0
        self.add_types = NULL
        self.add_depths = NULL

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
