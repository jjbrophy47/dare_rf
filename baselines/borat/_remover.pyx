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
from ._utils cimport split_samples
from ._utils cimport convert_int_ndarray
from ._utils cimport copy_int_array
from ._utils cimport dealloc

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
                  bint use_gini):
        """
        Constructor.
        """
        self.manager = manager
        self.tree_builder = tree_builder
        self.use_gini = use_gini
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

        # Data containers
        cdef int** X = NULL
        cdef int* y = NULL
        self.manager.get_data(&X, &y)

        cdef int* samples = convert_int_ndarray(remove_indices)
        cdef int  n_samples = remove_indices.shape[0]

        # check if any sample has already been deleted
        cdef int result = self.manager.check_sample_validity(samples, n_samples)
        if result == -1:
            return -1

        # check if any node at any layer needs to retrain
        self._remove(&tree.root, X, y, samples, n_samples)

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

        cdef int topd = self.tree_builder.topd
        cdef int min_support = self.tree_builder.min_support

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

                # exact node, retrain
                elif result == 2:
                    self._add_removal_type(result, node.depth)
                    self._retrain(&node_ptr, X, y, samples, n_samples)
                    free(samples)

                # update and recurse
                elif result == 0:

                    self._update_decision_node(&node, &split)
                    free(samples)

                    # prevent sim mode from updating beyond the deletion point
                    if self.tree_builder.sim_mode and node.depth == self.tree_builder.sim_depth:
                        free(split.left_indices)
                        free(split.right_indices)
                        return

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
        cdef Node* node = node_ptr[0]

        if self.tree_builder.sim_mode:
            self.tree_builder.sim_depth = node.depth
            return

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
        cdef Node* node = node_ptr[0]

        if self.tree_builder.sim_mode:
            self.tree_builder.sim_depth = node.depth
            return

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

        if self.tree_builder.sim_mode:
            self.tree_builder.sim_depth = node.depth
            self.retrain_sample_count += node.count - n_samples
            return

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
        self.tree_builder.features = copy_int_array(node.features, node.features_count)

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
        cdef Node* node = node_ptr[0]

        if self.tree_builder.sim_mode:
            return

        node.count = split.count
        node.pos_count = split.pos_count

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
        cdef bint use_gini = self.use_gini

        cdef double split_score = -1
        cdef double best_score = 1
        cdef int chosen_ndx = -1

        cdef int updated_count = node.count - n_samples
        cdef int updated_pos_count = node.pos_count - pos_count

        cdef int i
        cdef int j
        cdef int k

        cdef int topd = self.tree_builder.topd
        cdef int min_support = self.tree_builder.min_support

        cdef int result = 0

        if updated_pos_count > 0 and updated_pos_count < updated_count:

            # exact, check if best feature has changed
            if node.depth < topd or node.count < min_support:
                best_score = 1
                chosen_ndx = -1

                for j in range(node.features_count):

                    split_score = compute_split_score(use_gini, updated_count, node.left_counts[j],
                                                       node.right_counts[j], node.left_pos_counts[j],
                                                       node.right_pos_counts[j])

                    if split_score < best_score:
                        best_score = split_score
                        chosen_ndx = j

                # same feature is still best
                if node.feature == node.features[chosen_ndx]:
                    split_samples(node, X, y, samples, n_samples, split)
                    result = 0

                # new feature is best, retrain
                else:
                    result = 2

            # random node
            else:
                split_samples(node, X, y, samples, n_samples, split)
                result = 0

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
