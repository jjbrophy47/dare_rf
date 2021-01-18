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

        # counters
        cdef SIZE_t n_pos_samples = 0

        # update node counts
        n_pos_samples = self.update_node(&node, y, samples, n_samples)

        # leaf, check complete
        if node.is_leaf:
            self.update_leaf(&node, samples, n_samples)

        # decision node, but samples in same class, convert to leaf, check complete
        elif node.n_pos_samples == 0 or node.n_pos_samples == node.n_samples:
            self.convert_to_leaf(&node, samples, n_samples, &split)

        # decision node
        else:

            # update metadata
            n_usable_thresholds = self.update_metadata(&node, X, y, samples, n_samples)

            # no more usable thresholds, convert to leaf, check complete
            if n_usable_thresholds == 0:
                self.convert_to_leaf(&node, samples, n_samples, &split)

            # viable decision node
            else:

                # check optimal split
                result = self.check_optimal_split(node, X, y, samples, n_samples, pos_count, &split)

                # optimal split has changed, retrain, check complete
                if result == 1:
                    self.retrain(&node_ptr, X, y, samples, n_samples)

                # no retraining necessary, split samples and recurse
                else:

                    # split deleted samples and free original samples
                    split_samples(node, X, y, samples, n_samples, split)

                    # traverse left if any deleted samples go left
                    if split.n_left_samples > 0:
                        self._remove(&node.left, X, y, split.left_samples, split.n_left_samples)

                    # traverse right if any deleted samples go right
                    if split.n_right_samples > 0:
                        self._remove(&node.right, X, y, split.right_samples, split.n_right_samples)

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

        # return no. positive samples being deleted
        return n_pos_samples

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void update_leaf(self,
                          Node** node_ptr,
                          int*   samples,
                          int    n_samples) nogil:
        """
        Update leaf node properties: value and leaf_samples. Check complete.
        """
        cdef Node* node = node_ptr[0]

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

        # assign new leaf samples and value
        node.leaf_samples = leaf_samples

        # check complete: add deletion type and depth, and clean up
        self.add_removal_type(0, node.depth)
        free(samples)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _convert_to_leaf(self,
                               Node**       node_ptr,
                               SIZE_t*      samples,
                               SIZE_t       n_samples,
                               SplitRecord* split) nogil:
        """
        Convert decision node to a leaf node. Check complete.
        """
        cdef Node* node = node_ptr[0]

        #  updated leaf samples array
        cdef SIZE_t* leaf_samples = <SIZE_t *>malloc(node.n_samples * sizeof(SIZE_t))
        cdef SIZE_t  leaf_samples_count = 0

        # get leaf samples and remove deleted samples
        self.get_leaf_samples(node, samples, n_samples, &leaf_samples, &leaf_samples_count)

        # deallocate node / subtree
        dealloc(node)

        # set leaf properties
        node.is_leaf = True
        node.value = node.n_pos_samples / <DTYPE_t> node.n_samples
        node.leaf_samples = leaf_samples

        # reset decision node properties
        node.features = NULL
        node.n_features = 0
        node.chosen_feature = NULL
        node.chosen_threshold = NULL

        # reset children properties
        node.left = NULL
        node.right = NULL

        # check complete: add deletion type and depth, and clean up
        self.add_removal_type(0, node.depth)
        free(samples)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void retrain(self,
                      Node***   node_pp,
                      DTYPE_t** X,
                      INT32_t*  y,
                      SIZE_t*   samples,
                      SIZE_t    n_samples) nogil:
        """
        Rebuild subtree at this node.
        """
        cdef Node*  node = node_pp[0][0]
        cdef Node** node_ptr = node_pp[0]

        # updated leaf samples array
        cdef SIZE_t* leaf_samples = <SIZE_t *>malloc(node.n_samples * sizeof(SIZE_t))
        cdef SIZE_t  leaf_samples_count = 0

        # node properties
        cdef SIZE_t depth = node.depth
        cdef bint   is_left = node.is_left

        # get updated list of samples
        self.get_leaf_samples(node, samples, n_samples, &leaf_samples, &leaf_samples_count)

        # free node / subtree
        dealloc(node)
        free(node)

        # retrain node / subtree
        node_ptr[0] = self.tree_builder._build(X, y, leaf_samples, leaf_samples_count, depth, is_left)

        # check complete: add deletion type and depth, and clean up
        self.add_removal_type(1, node.depth)
        free(samples)

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
        cdef bint   add_sample = 1
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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int check_optimal_split(self,
                                 Node* node) nogil:
        """
        Compare all feature thresholds against the chosen feature / threshold.
        """

        # parameters
        cdef bint   use_gini = self.use_gini
        cdef SIZE_t topd = self.tree_builder.topd

        # keep track of the best feature / threshold
        cdef DTYPE_t best_score = 1000000
        cdef DTYPE_t split_score = -1

        # record the best feature / threshold
        cdef INT32_t chosen_feature_ndx = 0
        cdef INT32_t chosen_threshold_ndx = 0

        # object pointers
        cdef Feature*   feature = NULL
        cdef Threshold* threshold = NULL

        # counters
        cdef SIZE_t i = 0
        cdef SIZE_t j = 0
        cdef SIZE_t k = 0

        # return 1 if retraining is necessary, 0 otherwise
        cdef INT32_t result = 0

        # greedy node, check if the best feature / threshold has changed
        if node.depth >= topd:
            best_score = 1000000

            # get thresholds for each feature
            for j in range(node.n_features):
                feature = node.features[j]

                # compute split score for each threshold
                for k in range(feature.n_thresholds):
                    threshold = feature.thresholds[k]

                    # compute split score, entropy or Gini index
                    split_score = compute_split_score(use_gini,
                                                      node.n_samples,
                                                      threshold.n_left_samples,
                                                      threshold.n_right_samples,
                                                      threshold.n_left_pos_samples,
                                                      threshold.n_right_pos_samples)

                    # printf('[SN] threshold.value: %.3f, split_score: %.3f\n', threshold.value, split_score)

                    # keep threshold with the best score
                    if split_score < best_score:
                        best_score = split_score
                        chosen_feature_ndx = feature.index
                        chosen_threshold_value = threshold.value

            # check to see if the same feature / threshold is still the best
            result = (node.chosen_feature.index == chosen_feature_ndx and
                      node.chosen_threshold.value == chosen_threshold_value)

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

        # counters
        cdef int i

        # object pointers
        cdef Feature*   feature = NULL
        cdef Threshold* threshold = NULL

        # 2d array of indices designating invalid feautures / thresholds
        cdef SIZE_t* thresholds = NULL
        cdef SIZE_t  n_thresholds = 0

        # no. of viable thresholds at this feature
        cdef n_usable_thresholds = 0

        # update statistics for each feature
        for j in range(node.n_features):
            feature = node.features[j]

            # array holding indices of invalid thresholds for this feature
            thresholds = <SIZE_t *>malloc(feature.n_thresholds * sizeof(SIZE_t))
            n_thresholds = 0

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
                        thresholds[j][n_thresholds[j]] = k
                        n_thresholds[j] += 1

                    # viable threshold
                    else:
                        n_usable_thresholds += 1

            # if n_thresholds > 0

                # sort feature values and get list of candidate thresholds

                # sample a new threshold uniformly at random

                # make sure new threshold is not already being used by this feature

                # if viable threshold

                    # replace old threshold with this new threshold

                    # increment n_usable_thresholds

            # free thresholds array

        return n_usable_thresholds



    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void add_removal_type(self,
                               SIZE_t remove_type,
                               SIZE_t remove_depth) nogil:
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

    cdef np.ndarray _get_int_ndarray(self,
                                     INT32_t* data,
                                     SIZE_t   n_elem):
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
