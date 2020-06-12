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

from ._utils cimport compute_split_score
from ._utils cimport generate_distribution
from ._utils cimport find_max_divergence
from ._utils cimport copy_int_array
from ._utils cimport dealloc

cdef int UNDEF = -1

# =====================================
# Adder
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
                  double epsilon, double lmbda, bint use_gini):
        """
        Constructor.
        """
        self.manager = manager
        self.tree_builder = tree_builder
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.use_gini = use_gini
        self.min_samples_leaf = tree_builder.min_samples_leaf
        self.min_samples_split = tree_builder.min_samples_split

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

        cdef int min_retrain_layer = -1

        self.manager.get_data(&X, &y)
        self._resize_metrics(n_samples)

        self._detect_retrains(&tree.root, X, y, samples, n_samples, &min_retrain_layer)
        if min_retrain_layer >= 0:
            # printf("retraining needed at layer %d\n", min_retrain_layer)
            self._add_add_type(2, min_retrain_layer)
            self._retrain_min_layer(&tree.root, X, y, samples, n_samples, min_retrain_layer)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _detect_retrains(self, Node** node_ptr, int** X, int* y,
                                int* samples, int n_samples,
                                int* min_retrain_layer) nogil:
        """
        Recursively add the samples to this subtree.
        """

        cdef Node *node = node_ptr[0]

        cdef SplitRecord split
        cdef int result = 0

        cdef bint is_bottom_leaf = node.is_leaf and not node.features

        cdef int pos_count = 0

        if min_retrain_layer[0] != -1 and node.depth >= min_retrain_layer[0]:
            return

        for i in range(n_samples):
            if y[samples[i]] == 1:
                pos_count += 1

        if is_bottom_leaf:
            self._update_leaf(&node, y, samples, n_samples, pos_count)
            self._add_add_type(result, node.depth)

        else:
            self._update_splits(&node, X, y, samples, n_samples, pos_count)

            result = self._check_node(node, X, y, samples, n_samples,
                                      pos_count, &split)

            # retrain
            if result > 0:

                if min_retrain_layer[0] == -1 or node.depth < min_retrain_layer[0]:
                    min_retrain_layer[0] = node.depth

            else:

                if node.is_leaf:
                    self._update_leaf(&node, y, samples, n_samples, pos_count)
                    self._add_add_type(result, node.depth)

                # decision node
                else:
                    self._update_decision_node(&node, &split)

                    # traverse left
                    if split.left_count > 0:
                        self._detect_retrains(&node.left, X, y, split.left_indices,
                                              split.left_count, min_retrain_layer)

                    # traverse right
                    if split.right_count > 0:
                        self._detect_retrains(&node.right, X, y, split.right_indices,
                                              split.right_count, min_retrain_layer)

        if node.depth > 0 or min_retrain_layer[0] == -1:
            free(samples)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _retrain_min_layer(self, Node** node_ptr, int** X, int* y,
                                 int* samples, int n_samples,
                                 int min_retrain_layer) nogil:
        """
        Recursively find the nodes at the target layer and retrain them.
        """

        # printf('accessing node...\n')
        cdef Node* node = node_ptr[0]
        cdef SplitRecord split
        # printf('updating count...\n')
        cdef int updated_count = node.count + n_samples
        # printf('done updating count...\n')
        # printf('(count, n_samples, updated_count): (%d, %d, %d)\n', node.count, n_samples, updated_count)

        # printf('\nnode (depth, is_left, is_leaf, n_samples): (%d, %d, %d, %d)\n', node.depth, node.is_left, node.is_leaf, n_samples)

        # retrain
        if node.depth == min_retrain_layer:
            # printf('retraining...\n')

            if not node.is_leaf or (node.is_leaf and n_samples > 0):
                # printf('retraining...\n')

                # for i in range(n_samples):
                    # printf('samples[%d]: %d\n', i, samples[i])

                self._retrain(&node_ptr, X, y, samples, n_samples, updated_count)
                # printf('done retraining\n')

        else:
            # printf('splitting...\n')
            self._split_samples(node, X, y, samples, n_samples, &split)
            # printf('done splitting...\n')

            # for i in range(n_samples):
            #     printf('samples[%d]: %d\n', i, samples[i])

            # if split.left_indices == NULL or split.left_count == 0:
            #     printf('left indices is equal to NULL\n')

            # if split.right_indices == NULL or split.right_count == 0:
            #     printf('right indices is equal to NULL\n')

            # traverse left
            # printf('traversing left...\n')
            if node.left:
                self._retrain_min_layer(&node.left, X, y, split.left_indices,
                                        split.left_count, min_retrain_layer)

            # traverse right
            if node.right:
                self._retrain_min_layer(&node.right, X, y, split.right_indices,
                                        split.right_count, min_retrain_layer)

        # printf('freeing samples\n')
        free(samples)
        # printf('done freeing samples\n')

    # private
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _update_leaf(self, Node** node_ptr, int* y, int* samples,
                           int n_samples, int pos_count) nogil:
        """
        Update leaf node: count, pos_count, value, leaf_samples.
        """
        cdef Node* node = node_ptr[0]

        # add samples to leaf
        cdef int* leaf_samples = <int *>malloc((node.count + n_samples) * sizeof(int))
        cdef int  leaf_samples_count = 0
        self._get_leaf_samples(node, &leaf_samples, &leaf_samples_count)
        self._add_leaf_samples(samples, n_samples, &leaf_samples, &leaf_samples_count)
        free(node.leaf_samples)

        node.count += n_samples
        node.pos_count += pos_count
        node.value = node.pos_count / <double> node.count
        node.leaf_samples = leaf_samples

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _retrain(self, Node*** node_pp, int** X, int* y, int* samples,
                       int n_samples, int updated_count) nogil:
        """
        Rebuild subtree at this node.
        """
        cdef Node*  node = node_pp[0][0]
        cdef Node** node_ptr = node_pp[0]

        cdef int* leaf_samples = <int *>malloc(updated_count * sizeof(int))
        cdef int  leaf_samples_count = 0

        cdef int* rebuild_features = NULL

        cdef int depth = node.depth
        cdef int is_left = node.is_left
        cdef int features_count = node.features_count

        # printf('getting leaf samples...\n')
        self._get_leaf_samples(node, &leaf_samples, &leaf_samples_count)
        # for i in range(leaf_samples_count):
        #     printf('leaf_samples[%d]: %d\n', i, leaf_samples[i])

        # printf('adding leaf samples...\n')
        self._add_leaf_samples(samples, n_samples, &leaf_samples, &leaf_samples_count)
        # for i in range(leaf_samples_count):
        #     printf('leaf_samples[%d]: %d\n', i, leaf_samples[i])

        rebuild_features = copy_int_array(node.features, node.features_count)
        # printf('deallocing node...\n')
        dealloc(node)
        free(node)

        # for i in range(leaf_samples_count):
        #     printf('leaf_samples[%d]: %d\n', i, leaf_samples[i])

        # printf('rebuilding subtree...\n')
        node_ptr[0] = self.tree_builder._build(X, y, leaf_samples, leaf_samples_count,
                                               rebuild_features, features_count,
                                               depth, is_left, node.layer_budget_ptr)
        # printf('done rebuilding.\n')

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
                self._get_leaf_samples(node.left, leaf_samples_ptr, leaf_samples_count_ptr)

            if node.right:
                self._get_leaf_samples(node.right, leaf_samples_ptr, leaf_samples_count_ptr)

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
        Checks node for retraining and splits the add samples.
        """

        # parameters
        cdef double lmbda = self.lmbda
        cdef bint use_gini = self.use_gini

        cdef double* split_scores = NULL
        cdef double* distribution = NULL

        cdef int chosen_ndx = -1

        cdef double max_local_divergence

        cdef int updated_count = node.count + n_samples
        cdef int updated_pos_count = node.pos_count + pos_count

        cdef int i
        cdef int j
        cdef int k

        cdef int result = 0

        if updated_pos_count > 0 and updated_pos_count < updated_count:

            if node.sspd != NULL:

                split_scores = <double *>malloc(node.features_count * sizeof(double))
                distribution = <double *>malloc(node.features_count * sizeof(double))

                for j in range(node.features_count):

                    split_scores[j] = compute_split_score(use_gini, updated_count, node.left_counts[j],
                                                          node.right_counts[j], node.left_pos_counts[j],
                                                          node.right_pos_counts[j])

                    if node.features[j] == node.feature:
                        chosen_ndx = j

                # generate new probability and compare to previous probability
                generate_distribution(lmbda, &distribution, split_scores, node.features_count, updated_count)

                # remove affect of previous divergence and add affect of current divergence to layer budget
                max_local_divergence = find_max_divergence(node.sspd, distribution, node.features_count)
                node.layer_budget_ptr[0][node.depth] += node.divergence
                node.layer_budget_ptr[0][node.depth] -= max_local_divergence
                node.divergence = max_local_divergence

                if node.layer_budget_ptr[0][node.depth] >= 0:

                    # assign results from chosen feature
                    split.left_indices = <int *>malloc(n_samples * sizeof(int))
                    split.right_indices = <int *>malloc(n_samples * sizeof(int))
                    j = 0
                    k = 0
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

                # bounds exceeded => retrain
                else:
                    result = 2

                free(split_scores)
                free(distribution)

            # leaf now has samples => retrain
            else:
                result = 3

        # all samples in one class => leaf (should already be a leaf)
        else:
            result = 0

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

            node.left_counts[j] += left_count
            node.left_pos_counts[j] += left_pos_count
            node.right_counts[j] += (n_samples - left_count)
            node.right_pos_counts[j] += (pos_count - left_pos_count)

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
        Adds to the addition metrics.
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
