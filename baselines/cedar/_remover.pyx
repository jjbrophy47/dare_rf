"""
Data remover.
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

from ._utils cimport compute_split_score
from ._utils cimport generate_distribution
from ._utils cimport sample_distribution
from ._utils cimport find_max_divergence
from ._utils cimport convert_int_ndarray
from ._utils cimport copy_int_array
from ._utils cimport dealloc
from ._utils cimport UINT32_t

cdef int UNDEF = -1
cdef double UNDEF_LEAF_VAL = 0.5

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

    property retrain_sample_count:
        def __get__(self):
            return self.retrain_sample_count

    def __cinit__(self, _DataManager manager, _TreeBuilder tree_builder,
                  double lmbda, bint use_gini):
        """
        Constructor.
        """
        self.manager = manager
        self.tree_builder = tree_builder
        self.lmbda = lmbda
        self.use_gini = use_gini
        self.min_samples_leaf = tree_builder.min_samples_leaf
        self.min_samples_split = tree_builder.min_samples_split

        self.sim_mode = 0
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
    cpdef int remove(self, _Tree tree, np.ndarray remove_indices, bint sim_mode):
        """
        Remove the data from remove_indices from the learned _Tree.
        """

        # Data containers
        cdef int** X = NULL
        cdef int* y = NULL
        self.manager.get_data(&X, &y)

        cdef int* samples = convert_int_ndarray(remove_indices)
        cdef int  n_samples = remove_indices.shape[0]

        cdef int  result = 0

        # check if any sample has already been deleted
        result = self.manager.check_sample_validity(samples, n_samples)
        if result == -1:
            return -1

        # check if any node at any layer needs to retrain
        self.sim_mode = sim_mode
        self._remove(&tree.root, X, y, samples, n_samples)
        self.sim_mode = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _remove(self, Node** node_ptr, int** X, int* y,
                      int* samples, int n_samples) nogil:
        """
        Recursively update and detect any layers that need retraining.
        """

        cdef Node *node = node_ptr[0]

        cdef SplitRecord split
        cdef int result = 0
        cdef int resampled = 0

        cdef UINT32_t* random_state = &self.tree_builder.rand_r_state
        cdef int chosen_ndx = -1

        cdef int topd = self.tree_builder.topd
        cdef int min_support = self.tree_builder.min_support
        cdef double lmbda = self.lmbda

        cdef bint is_bottom_leaf = node.is_leaf and not node.features

        cdef int i = 0
        cdef int pos_count = 0

        # compute positive count
        for i in range(n_samples):
            if y[samples[i]] == 1:
                pos_count += 1

        # bottom node
        if is_bottom_leaf:
            self._add_removal_type(result, node.depth)
            self._update_leaf(&node, y, samples, n_samples, pos_count)
            free(samples)

        # middle node
        else:
            self._update_splits(&node, X, y, samples, n_samples, pos_count)

            # leaf node
            if node.is_leaf:
                self._add_removal_type(result, node.depth)
                self._update_leaf(&node, y, samples, n_samples, pos_count)
                free(samples)

            # decision node
            else:
                result = self._check_node(node, X, y, samples, n_samples,
                                          pos_count, &split)

                # convert to leaf
                if result == 1:
                    self._add_removal_type(result, node.depth)
                    self._convert_to_leaf(&node, samples, n_samples, &split)
                    free(samples)

                # retrain
                elif result == 2:

                    if self.sim_mode:
                        self.retrain_sample_count += node.count
                        return

                    # exact node, ONLY retrain the node
                    if lmbda == -1 or node.depth >= topd or node.count < min_support:
                        self._add_removal_type(result, node.depth)
                        self._retrain(&node_ptr, X, y, samples, n_samples)
                        free(samples)

                    # semi-random node, retrain
                    else:
                        chosen_ndx = sample_distribution(node.sspd, node.features_count, random_state)

                        # feature resampled, recurse
                        if node.features[chosen_ndx] == node.feature:
                            self._split_samples(node, X, y, samples, n_samples, &split)
                            resampled = 1

                        # retrain
                        else:
                            self._add_removal_type(result, node.depth)
                            self._retrain(&node_ptr, X, y, samples, n_samples)
                            free(samples)

                # update and recurse
                if result == 0 or resampled:

                    self._update_decision_node(&node, &split)
                    free(samples)

                    # traverse left
                    if split.left_count > 0:
                        self._remove(&node.left, X, y, split.left_indices, split.left_count)
                    else:
                        free(split.left_indices)

                    # traverse right
                    if split.right_count > 0:
                        self._remove(&node.right, X, y, split.right_indices, split.right_count)
                    else:
                        free(split.right_indices)

    # private
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _update_leaf(self, Node** node_ptr, int* y, int* samples, int n_samples,
                           int pos_count) nogil:
        """
        Update leaf node: count, pos_count, value, leaf_samples.
        """
        if self.sim_mode:
            return

        cdef Node* node = node_ptr[0]

        # remove samples from leaf
        cdef int* leaf_samples = <int *>malloc((node.count - n_samples) * sizeof(int))
        cdef int  leaf_samples_count = 0
        self._get_leaf_samples(node, samples, n_samples, &leaf_samples, &leaf_samples_count)
        free(node.leaf_samples)

        node.count -= n_samples
        node.pos_count -= pos_count
        node.leaf_samples = leaf_samples
        if node.count > 0:
            node.value = node.pos_count / <double> node.count
        else:
            node.value = UNDEF_LEAF_VAL

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _convert_to_leaf(self, Node** node_ptr, int* samples, int n_samples,
                               SplitRecord *split) nogil:
        """
        Convert decision node to a leaf node.
        """
        if self.sim_mode:
            return

        cdef Node* node = node_ptr[0]

        cdef int* leaf_samples = <int *>malloc(split.count * sizeof(int))
        cdef int  leaf_samples_count = 0
        self._get_leaf_samples(node, samples, n_samples, &leaf_samples, &leaf_samples_count)
        leaf_samples = <int *>realloc(leaf_samples, leaf_samples_count * sizeof(int))

        dealloc(node.left)
        dealloc(node.right)
        free(node.leaf_samples)

        node.count = split.count
        node.pos_count = split.pos_count

        node.is_leaf = 1
        node.value = node.pos_count / <double> node.count
        node.leaf_samples = leaf_samples

        node.sspd = NULL
        node.feature = UNDEF

        node.left = NULL
        node.right = NULL

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _retrain(self, Node*** node_pp, int** X, int* y, int* samples,
                       int n_samples) nogil:
        """
        Rebuild subtree at this node.
        """
        cdef Node*  node = node_pp[0][0]
        cdef Node** node_ptr = node_pp[0]

        cdef int* leaf_samples = <int *>malloc(node.count * sizeof(int))
        cdef int  leaf_samples_count = 0

        cdef int depth = node.depth
        cdef int is_left = node.is_left

        cdef int* invalid_features = NULL
        cdef int  invalid_features_count = node.invalid_features_count

        cdef int i

        self._get_leaf_samples(node, samples, n_samples,
                               &leaf_samples, &leaf_samples_count)
        leaf_samples = <int *>realloc(leaf_samples, leaf_samples_count * sizeof(int))

        self.retrain_sample_count += leaf_samples_count

        invalid_features = copy_int_array(node.invalid_features, node.invalid_features_count)
        dealloc(node)
        free(node)

        node_ptr[0] = self.tree_builder._build(X, y, leaf_samples, leaf_samples_count,
                                               invalid_features, invalid_features_count,
                                               depth, is_left)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _get_leaf_samples(self, Node* node, int* remove_samples, int n_remove_samples,
                                int** leaf_samples_ptr, int* leaf_samples_count_ptr) nogil:
        """
        Recursively obtain and filter the samples at the leaves.
        """
        cdef bint add_sample
        cdef int i
        cdef int j

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
        if self.sim_mode:
            return

        cdef Node* node = node_ptr[0]
        node.count = split.count
        node.pos_count = split.pos_count

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _split_samples(self, Node* node, int** X, int* y,
                             int* samples, int n_samples,
                             SplitRecord *split) nogil:
        """
        Split samples based on the chosen feature.
        """

        cdef int j = 0
        cdef int k = 0

        # assign results from chosen feature
        split.left_indices = <int *>malloc(n_samples * sizeof(int))
        split.right_indices = <int *>malloc(n_samples * sizeof(int))
        for i in range(n_samples):
            if X[samples[i]][node.feature] == 1:
                split.left_indices[j] = samples[i]
                j += 1
            else:
                split.right_indices[k] = samples[i]
                k += 1
        split.left_indices = <int *>realloc(split.left_indices, j * sizeof(int))
        split.right_indices = <int *>realloc(split.right_indices, k * sizeof(int))
        split.left_count = j
        split.right_count = k

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int _check_node(self, Node* node, int** X, int* y,
                          int* samples, int n_samples, int pos_count,
                          SplitRecord *split) nogil:
        """
        Checks node for retraining and splits the removal samples.
        """

        # parameters
        cdef double lmbda = self.lmbda
        cdef bint use_gini = self.use_gini

        cdef double* split_scores = NULL
        cdef double* distribution = NULL

        cdef int chosen_ndx = -1

        cdef double divergence = 0

        cdef int updated_count = node.count - n_samples
        cdef int updated_pos_count = node.pos_count - pos_count

        cdef int i
        cdef int j
        cdef int k

        cdef int topd = self.tree_builder.topd
        cdef int min_support = self.tree_builder.min_support

        cdef int result = 0

        if updated_pos_count > 0 and updated_pos_count < updated_count:

            split_scores = <double *>malloc(node.features_count * sizeof(double))
            distribution = <double *>malloc(node.features_count * sizeof(double))

            for j in range(node.features_count):

                split_scores[j] = compute_split_score(use_gini, updated_count, node.left_counts[j],
                                                      node.right_counts[j], node.left_pos_counts[j],
                                                      node.right_pos_counts[j])

                if node.features[j] == node.feature:
                    chosen_ndx = j

            # set to exact if node requirements not set
            if lmbda > 0 and (node.depth >= topd or node.count < min_support):
                lmbda = -1

            # generate new probability and compare to previous probability
            generate_distribution(lmbda, &distribution, split_scores,
                                  node.features_count, updated_count, use_gini)

            # remove affect of previous divergence and add affect of current divergence to layer budget
            divergence = find_max_divergence(node.sspd, distribution, node.features_count)

            # divergence within the allowable budget
            if divergence <= node.budget:
                self._split_samples(node, X, y, samples, n_samples, split)
                free(distribution)

            # bounds exceeded => retrain
            else:
                free(node.sspd)
                node.sspd = distribution
                result = 2

            free(split_scores)

        # all samples in one class => leaf
        else:
            result = 1

        split.count = updated_count
        split.pos_count = updated_pos_count

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _update_splits(self, Node** node_ptr, int** X, int* y,
                            int* samples, int n_samples, int pos_count) nogil:
        """
        Update the metadata of this node.
        """
        cdef Node* node = node_ptr[0]

        cdef int left_count
        cdef int left_pos_count

        cdef int i

        # compute statistics for each attribute
        for j in range(node.features_count):

            left_count = 0
            left_pos_count = 0

            for i in range(n_samples):

                if X[samples[i]][node.features[j]] == 1:
                    left_count += 1
                    left_pos_count += y[samples[i]]

            node.left_counts[j] -= left_count
            node.left_pos_counts[j] -= left_pos_count
            node.right_counts[j] -= (n_samples - left_count)
            node.right_pos_counts[j] -= (pos_count - left_pos_count)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _add_removal_type(self, int remove_type, int remove_depth) nogil:
        """
        Adds to the removal metrics.
        """
        if self.remove_types and self.remove_depths:
            if self.remove_count + 1 == self.capacity:
                self.capacity *= 2

            self.remove_types = <int *>realloc(self.remove_types, self.capacity * sizeof(int))
            self.remove_depths = <int *>realloc(self.remove_depths, self.capacity * sizeof(int))

        else:
            self.capacity = 10
            self.remove_types = <int *>malloc(self.capacity * sizeof(int))
            self.remove_depths = <int *>malloc(self.capacity * sizeof(int))

        self.remove_types[self.remove_count] = remove_type
        self.remove_depths[self.remove_count] = remove_depth
        self.remove_count += 1

    cpdef void clear_remove_metrics(self):
        """
        Resets deletion statistics.
        """
        free(self.remove_types)
        free(self.remove_depths)
        self.remove_count = 0
        self.remove_types = NULL
        self.remove_depths = NULL
        self.retrain_sample_count = 0

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