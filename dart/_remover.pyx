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

    def __cinit__(self,
                  _DataManager manager,
                  _TreeBuilder tree_builder,
                  bint         use_gini):
        """
        Constructor.
        """
        self.manager = manager
        self.tree_builder = tree_builder
        self.use_gini = use_gini
        self.min_samples_leaf = tree_builder.min_samples_leaf
        self.min_samples_split = tree_builder.min_samples_split

        # initialize metric properties
        self.capacity = 10
        self.remove_count = 0
        self.remove_types = <SIZE_t *>malloc(self.capacity * sizeof(SIZE_t))
        self.remove_depths = <SIZE_t *>malloc(self.capacity * sizeof(SIZE_t))

    def __dealloc__(self):
        """
        Destructor.
        """
        # free removal types
        if self.remove_types:
            free(self.remove_types)

        # free removal depths
        if self.remove_depths:
            free(self.remove_depths)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef INT32_t remove(self, _Tree tree, np.ndarray remove_indices):
        """
        Remove the data specified by the `remove_indices` from the
        learned _Tree.
        """

        # Data containers
        cdef DTYPE_t** X = NULL
        cdef INT32_t*  y = NULL
        self.manager.get_data(&X, &y)

        cdef SIZE_t* samples = convert_int_ndarray(remove_indices)
        cdef SIZE_t  n_samples = remove_indices.shape[0]

        # check if any sample has already been deleted
        cdef INT32_t result = self.manager.check_sample_validity(samples, n_samples)
        if result == -1:
            return -1

        # recurse through the tree and retrain nodes / substrees as necessary
        self._remove(&tree.root, X, y, samples, n_samples)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _remove(self,
                      Node**    node_ptr,
                      DTYPE_t** X,
                      INT32_t*  y,
                      SIZE_t*   samples,
                      SIZE_t    n_samples) nogil:
        """
        Update and retrain this node if necessary, otherwise traverse
        to its children.
        """
        cdef Node *node = node_ptr[0]

        # result containers
        cdef SplitRecord split
        cdef INT32_t     result = 0

        # object properties
        cdef SIZE_t topd = self.tree_builder.topd

        # boolean variables
        # cdef bint is_bottom_leaf = node.is_leaf and not node.features

        # counters
        # cdef SIZE_t i = 0
        cdef SIZE_t n_pos_samples = 0

        # update node counts
        n_pos_samples = self.update_node(&node, y, samples, n_samples)

        # leaf
        if node.is_leaf:
            self.update_leaf(&node, samples, n_samples)
            self.add_removal_type(result, node.depth)
            free(samples)

        # decision node, but all samples are now in the same class, convert to leaf
        elif node.n_pos_samples == 0 or node.n_pos_samples == node.n_samples:
            pass

        # decision node
        else:

            # update metadata
            self.update_metadata(&node, X, y, samples, n_samples)

            # if no usable thresholds
                # convert to leaf

            # check optimal split

            # if different split is optimal
                # retrain

            # else
                # traverse left
                # traverse right


        # # bottom node
        # if is_bottom_leaf:
        #     self._add_removal_type(result, node.depth)
        #     self._update_leaf(&node, y, samples, n_samples, pos_count)
        #     free(samples)

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
                        free(split.left_samples)
                        free(split.right_samples)
                        return

                    # traverse left
                    if split.n_left_samples > 0:
                        self._remove(&node.left, X, y, split.left_samples, split.n_left_samples)
                    else:
                        free(split.left_samples)

                    # traverse right
                    if split.n_right_samples > 0:
                        self._remove(&node.right, X, y, split.right_samples, split.n_right_samples)
                    else:
                        free(split.right_samples)

    # private
    cdef SIZE_t update_node(self,
                            Node**   node_ptr,
                            INT32_t* y,
                            SIZE_t*  samples,
                            SIZE_t   n_samples) nogil:
        """
        Update node counts based on the `samples` being deleted.
        """
        cdef Node *node = node_ptr[0]

        # compute number of positive samples being deleted
        cdef SIZE_t n_pos_samples = 0
        for i in range(n_samples):
            n_pos_samples += y[samples[i]]

        # update node counts
        cdef Node *node = <Node *>malloc(sizeof(Node))
        node.n_samples -= n_samples
        node.n_pos_samples -= n_pos_samples

        # 
        if updated_pos_count > 0 and updated_pos_count < updated_count:

        # return the number of positive samples being deleted
        return n_pos_samples

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void update_leaf(self,
                          Node** node_ptr,
                          int*   samples,
                          int    n_samples) nogil:
        """
        Update leaf node properties: value and leaf_samples.
        """
        cdef Node* node = node_ptr[0]

        # if self.tree_builder.sim_mode:
        #     self.tree_builder.sim_depth = node.depth
        #     return

        # remove deleted samples from leaf
        # cdef SIZE_t* leaf_samples = <SIZE_t *>malloc((node.n_samples - n_samples) * sizeof(SIZE_t))

        # update leaf value
        if node.n_samples > 0:
            node.value = node.n_pos_samples / <double> node.n_samples
        else:
            node.value = UNDEF_LEAF_VAL

        # update leaf samples array
        cdef SIZE_t* leaf_samples = <SIZE_t *>malloc((node.n_samples) * sizeof(SIZE_t))
        cdef SIZE_t  leaf_samples_count = 0

        # remove deleted samples from leaf
        self.get_leaf_samples(node, samples, n_samples, &leaf_samples, &leaf_samples_count)

        # free old leaf samples array
        free(node.leaf_samples)

        # node.n_samples -= n_samples
        # node.n_pos_samples -= pos_count

        # assign new leaf samples and value
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

        if self.tree_builder.sim_mode:
            self.tree_builder.sim_depth = node.depth
            return

        # cdef int* leaf_samples = <int *>malloc(split.n_samples * sizeof(int))
        cdef int* leaf_samples = NULL
        cdef int  leaf_samples_count = 0
        self._get_leaf_samples(node, samples, n_samples, &leaf_samples, &leaf_samples_count)
        leaf_samples = <int *>realloc(leaf_samples, leaf_samples_count * sizeof(int))

        dealloc(node.left)
        dealloc(node.right)
        free(node.leaf_samples)

        # node.n_samples = split.n_samples
        # node.n_pos_samples = split.n_pos_samples

        node.is_leaf = 1
        node.value = node.n_pos_samples / <double> node.n_samples
        node.leaf_samples = leaf_samples

        node.chosen_feature = NULL

        node.left = NULL
        node.right = NULL

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _retrain(self, Node*** node_pp, double** X, int* y, int* samples,
                       int n_samples) nogil:
        """
        Rebuild subtree at this node.
        """
        cdef Node*  node = node_pp[0][0]

        if self.tree_builder.sim_mode:
            self.tree_builder.sim_depth = node.depth
            self.retrain_sample_count += node.n_samples - n_samples
            return

        cdef Node** node_ptr = node_pp[0]

        cdef int* leaf_samples = <int *>malloc(node.n_samples * sizeof(int))
        cdef int  leaf_samples_count = 0

        cdef int depth = node.depth
        cdef int is_left = node.is_left

        # cdef int* invalid_features = NULL
        # cdef int  invalid_features_count = node.invalid_features_count

        self._get_leaf_samples(node, samples, n_samples,
                               &leaf_samples, &leaf_samples_count)
        leaf_samples = <int *>realloc(leaf_samples, leaf_samples_count * sizeof(int))

        self.retrain_sample_count += leaf_samples_count

        # invalid_features = copy_int_array(node.invalid_features, node.invalid_features_count)
        # self.tree_builder.features = copy_int_array(node.features, node.features_count)

        dealloc(node)
        free(node)

        node_ptr[0] = self.tree_builder._build(X, y, leaf_samples, leaf_samples_count,
                                               # invalid_features, invalid_features_count,
                                               depth, is_left)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void get_leaf_samples(self,
                               Node*    node,
                               SIZE_t*  remove_samples,
                               SIZE_t   n_remove_samples,
                               SIZE_t** leaf_samples_ptr,
                               SIZE_t*  leaf_samples_count_ptr) nogil:
        """
        Recursively obtain the samples at the leaves and filter out
        deleted samples.
        """
        cdef bint add_sample = 1
        cdef SIZE_t i = 0
        cdef SIZE_t j = 0

        # leaf
        if node.is_leaf:

            # loop through all samples at this leaf
            for i in range(node.n_samples):
                add_sample = 1

                # loop through all deleted samples
                for j in range(n_remove_samples):

                    # do not add sample to results if it has been deleted
                    if node.leaf_samples[i] == remove_samples[j]:
                        add_sample = 0
                        break

                # add sample to results if it has not been deleted
                if add_sample:
                    leaf_samples_ptr[0][leaf_samples_count_ptr[0]] = node.leaf_samples[i]
                    leaf_samples_count_ptr[0] += 1

        # decision node
        else:

            # traverse left
            if node.left:
                self.get_leaf_samples(node.left,
                                      remove_samples,
                                      n_remove_samples,
                                      leaf_samples_ptr,
                                      leaf_samples_count_ptr)

            # traverse right
            if node.right:
                self.get_leaf_samples(node.right,
                                      remove_samples,
                                      n_remove_samples,
                                      leaf_samples_ptr,
                                      leaf_samples_count_ptr)

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # cdef void _update_decision_node(self, Node** node_ptr, SplitRecord *split) nogil:
    #     """
    #     Update tree with node metadata.
    #     """
    #     if self.tree_builder.sim_mode:
    #         return

    #     cdef Node* node = node_ptr[0]
        # node.n_samples = split.n_samples
        # node.n_pos_samples = split.n_pos_samples

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int _check_node(self, Node* node, double** X, int* y,
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

        cdef int updated_count = node.n_samples - n_samples
        cdef int updated_pos_count = node.n_pos_samples - pos_count

        cdef int i
        cdef int j
        cdef int k

        cdef int topd = self.tree_builder.topd
        # cdef int min_support = self.tree_builder.min_support

        cdef int result = 0

        if updated_pos_count > 0 and updated_pos_count < updated_count:

            # exact, check if best feature has changed
            if node.depth >= topd:
                best_score = 1
                chosen_ndx = -1

                for j in range(node.n_features):

                    # split_score = compute_split_score(use_gini, updated_count, node.left_counts[j],
                    #                                    node.right_counts[j], node.left_pos_counts[j],
                    #                                    node.right_pos_counts[j])

                    # TODO
                    split_score = 1.0

                    if split_score < best_score:
                        best_score = split_score
                        chosen_ndx = j

                # same feature is still best
                if node.chosen_feature.index == node.features[chosen_ndx].index:
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

        # split.n_samples = updated_count
        # split.n_pos_samples = updated_pos_count

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef SIZE_t update_metadata(self,
                                Node**    node_ptr,
                                DTYPE_t** X,
                                INT32_t*  y,
                                SIZE_t*   samples,
                                SIZE_t    n_samples) nogil:
        """
        Update each threshold for all features at this node.
        """
        cdef Node* node = node_ptr[0]

        cdef int left_count
        cdef int left_pos_count

        cdef int i

        cdef Feature*   feature = NULL
        cdef Threshold* threshold = NULL

        # update statistics for each feature
        for j in range(node.n_features):
            feature = node.features[j]

            # update statistics for each threshold in this feature
            for k in range(feature.n_thresholds):
                threshold = feature.thresholds[k]

                # loop through each deleted sample
                for i in range(n_samples):

                    # decrement left branch of this threshold
                    if X[samples[i]][feature.index] <= threshold.value:
                        threshold.n_left_samples -= 1
                        threshold.n_left_pos_samples -= y[samples[i]]

                    # decrement right branch of this threshold
                    else:
                        threshold.n_right_samples -= 1
                        threshold.n_right_pos_samples -= y[samples[i]]

                    # decrement left value of this threshold
                    if X[samples[i]][feature.index] == threshold.v1:
                        threshold.n_v1_samples -= 1
                        threshold.n_v1_pos_samples -= 1

                    # decrement right value of this threshold
                    elif X[samples[i]][feature.index] == threshold.v2:
                        threshold.n_v2_samples -= 1
                        threshold.n_v2_pos_samples -= 1

                    # check to see if this threshold is invalid
                    if threshold.n_left_samples == 0 or threshold.n_right_samples == 0 or ratio_stuff:
                        # TODO: flag this threshold to get rid of
                        break



    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _add_removal_type(self, SIZE_t remove_type, SIZE_t remove_depth) nogil:
        """
        Add type and depth to the removal metrics.
        """
        if self.remove_types and self.remove_depths:
            if self.remove_count + 1 == self.capacity:
                self.capacity *= 2

            self.remove_types = <SIZE_t *>realloc(self.remove_types, self.capacity * sizeof(SIZE_t))
            self.remove_depths = <SIZE_t *>realloc(self.remove_depths, self.capacity * sizeof(SIZE_t))

        else:
            self.capacity = 10
            self.remove_types = <SIZE_t *>malloc(self.capacity * sizeof(SIZE_t))
            self.remove_depths = <SIZE_t *>malloc(self.capacity * sizeof(SIZE_t))

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

    cdef np.ndarray _get_int_ndarray(self, INT32_t *data, SIZE_t n_elem):
        """
        Wraps value as a 1-d NumPy array.
        The array keeps a reference to this Tree, which manages the underlying memory.
        """
        cdef SIZE_t shape[1]
        shape[0] = n_elem
        cdef np.ndarray arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, data)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr
