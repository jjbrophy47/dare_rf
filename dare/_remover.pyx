# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
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

from ._tree cimport Feature
from ._tree cimport Threshold
from ._splitter cimport FEATURE_THRESHOLD
from ._splitter cimport get_candidate_thresholds

from ._utils cimport rand_int
from ._utils cimport rand_uniform
from ._utils cimport compute_split_score
from ._utils cimport split_samples

from ._utils cimport create_feature
from ._utils cimport copy_feature
from ._utils cimport free_feature
from ._utils cimport free_features

from ._utils cimport copy_threshold
from ._utils cimport free_thresholds

from ._utils cimport create_intlist
from ._utils cimport copy_intlist
from ._utils cimport free_intlist

from ._utils cimport copy_indices
from ._utils cimport convert_int_ndarray
from ._utils cimport copy_int_array

from libc.math cimport fabs

from ._utils cimport dealloc
from ._argsort cimport sort

cdef INT32_t UNDEF = -1
cdef DTYPE_t UNDEF_LEAF_VAL = 0.5

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
            return self.get_int_ndarray(self.remove_types, self.remove_count)

    property remove_depths:
        def __get__(self):
            return self.get_int_ndarray(self.remove_depths, self.remove_count)

    property remove_costs:
        def __get__(self):
            return self.get_int_ndarray(self.remove_costs, self.remove_count)

    def __cinit__(self,
                  _DataManager manager,
                  _TreeBuilder tree_builder,
                  _Config      config):
        """
        Constructor.
        """
        self.manager = manager
        self.tree_builder = tree_builder
        self.config = config

        # initialize metric properties
        self.capacity = 10
        self.remove_count = 0
        self.remove_types = <INT32_t *>malloc(self.capacity * sizeof(INT32_t))
        self.remove_depths = <INT32_t *>malloc(self.capacity * sizeof(INT32_t))
        self.remove_costs = <INT32_t *>malloc(self.capacity * sizeof(INT32_t))

    def __dealloc__(self):
        """
        Destructor.
        """
        # printf('[R - D] dealloc\n')
        # free removal types
        if self.remove_types:
            free(self.remove_types)

        # free removal depths
        if self.remove_depths:
            free(self.remove_depths)

    cpdef INT32_t remove(self, _Tree tree, np.ndarray remove_indices):
        """
        Remove the data specified by the `remove_indices` from the
        learned _Tree.
        """

        # Data containers
        cdef DTYPE_t** X = NULL
        cdef INT32_t*  y = NULL
        self.manager.get_data(&X, &y)

        # convert the remove indices ndarray to an IntList object
        cdef IntList* remove_samples = <IntList *>malloc(sizeof(IntList))
        remove_samples.arr = convert_int_ndarray(remove_indices)
        remove_samples.n = remove_indices.shape[0]

        # check if any sample has already been deleted
        cdef INT32_t result = self.manager.check_remove_samples_validity(remove_samples)
        if result == -1:
            return -1

        # recurse through the tree and retrain nodes / subtrees as necessary
        self._remove(&tree.root, X, y, remove_samples)

    cdef void _remove(self,
                      Node**    node_ptr,
                      DTYPE_t** X,
                      INT32_t*  y,
                      IntList*  remove_samples) nogil:
        """
        Update and retrain this node if necessary, otherwise traverse
        to its children.
        """
        cdef Node* node = node_ptr[0]

        # result containers
        cdef SplitRecord split
        cdef SIZE_t      n_usable_thresholds = 0
        cdef INT32_t     result = 0

        # printf('\n[R] n_remove_samples: %ld, depth: %ld, is_left: %d\n', remove_samples.n, node.depth, node.is_left)

        # update node counts
        self.update_node(node, y, remove_samples)

        # leaf, check complete
        if node.is_leaf:
            # printf('[R] update leaf\n')
            self.update_leaf(node, remove_samples)
            # printf('[R] leaf.value: %.2f\n', node.value)

        # decision node, but samples in same class, convert to leaf, check complete
        elif node.n_pos_samples == 0 or node.n_pos_samples == node.n_samples:
            # printf('[R] convert to leaf\n')
            self.convert_to_leaf(node, remove_samples)
            # printf('[R] convert to leaf, leaf.value: %.2f\n', node.value)

        # decision node
        else:

            # update metadata
            # printf('[R] update metadata\n')
            n_usable_thresholds = self.update_metadata(node, X, y, remove_samples)
            # printf('[R] n_usable_thresholds: %ld\n', n_usable_thresholds)

            # chosen feature / threshold is now invalid, check complete
            if n_usable_thresholds < 0:
                # printf('[R] chosen feature / threshold invalid, retrain\n')
                self.retrain(node_ptr, X, y, remove_samples)

            # greedy node in which there are no more usable thresholds, convert to leaf, check complete
            elif n_usable_thresholds == 0:
                # printf('[R] convert to leaf\n')
                self.convert_to_leaf(node, remove_samples)
                # printf('[R] convert to leaf, leaf.value: %.2f\n', node.value)

            # viable decision node
            else:

                # check if optimal feature / threshold has changed
                # printf('[R] check optimal split\n')
                result = self.check_optimal_split(node)

                # optimal split has changed, retrain, check complete
                if result == 1:
                    # printf('[R] optimal split has changed, retrain\n')
                    self.retrain(node_ptr, X, y, remove_samples)
                    # printf('[R] retrain, node.depth: %ld, node.n_samples: %ld, node.n_features: %ld\n',
                    #        node_ptr[0].depth, node_ptr[0].n_samples, node_ptr[0].n_features)

                # no retraining necessary, split samples and recurse
                else:

                    # split deleted samples and free original deleted samples
                    # printf('[R] split samples\n')
                    split_samples(node, X, y, remove_samples, &split, 0)
                    # printf('[R] split samples, split.left_samples.n: %ld, split.right_samples.n: %ld\n',
                    #        split.left_samples.n, split.right_samples.n)

                    # traverse left if any deleted samples go left
                    if split.left_samples != NULL:
                        self._remove(&node.left, X, y, split.left_samples)

                    # traverse right if any deleted samples go right
                    if split.right_samples != NULL:
                        self._remove(&node.right, X, y, split.right_samples)

    # private
    cdef void update_node(self,
                          Node*    node,
                          INT32_t* y,
                          IntList* remove_samples) nogil:
        """
        Update node counts based on the `remove_samples` being deleted.
        """

        # compute number of positive samples being deleted
        cdef SIZE_t n_pos_remove_samples = 0
        for i in range(remove_samples.n):
            n_pos_remove_samples += y[remove_samples.arr[i]]

        # update node counts
        node.n_samples -= remove_samples.n
        node.n_pos_samples -= n_pos_remove_samples

    cdef void update_leaf(self,
                          Node*    node,
                          IntList* remove_samples) nogil:
        """
        Update leaf node properties: value and leaf_samples. Check complete.
        """

        # update leaf value
        if node.n_samples > 0:
            node.value = node.n_pos_samples / <double> node.n_samples

        # no samples left
        else:
            node.value = UNDEF_LEAF_VAL

        # update leaf samples array
        cdef IntList* leaf_samples = create_intlist(node.n_samples, 0)

        # remove deleted samples from leaf
        node.n_samples += remove_samples.n  # must check ALL leaf samples, even deleted ones
        get_leaf_samples(node, remove_samples, leaf_samples)
        node.n_samples -= remove_samples.n

        # free old leaf samples array
        free(node.leaf_samples)

        # assign new leaf samples and value
        node.leaf_samples = copy_indices(leaf_samples.arr, leaf_samples.n)

        # free leaf samples
        free_intlist(leaf_samples)

        # check complete: add deletion type and depth, and clean up
        self.add_metric(0, node.depth, 0)

        # free remove samples
        free_intlist(remove_samples)

    cdef void convert_to_leaf(self,
                              Node*        node,
                              IntList*     remove_samples) nogil:
        """
        Convert decision node to a leaf node. Check complete.
        """

        # get leaf samples and remove deleted samples
        cdef IntList* leaf_samples = create_intlist(node.n_samples, 0)
        get_leaf_samples(node, remove_samples, leaf_samples)

        # deallocate node / subtree
        dealloc(node)

        # set leaf properties
        node.is_leaf = True
        node.value = node.n_pos_samples / <DTYPE_t> node.n_samples
        node.leaf_samples = copy_indices(leaf_samples.arr, leaf_samples.n)

        # reset decision node properties
        node.features = NULL
        node.n_features = 0
        node.constant_features = NULL
        node.chosen_feature = NULL
        node.chosen_threshold = NULL

        # reset children properties
        node.left = NULL
        node.right = NULL

        # check complete: add deletion type and depth, and clean up
        self.add_metric(0, node.depth, 0)
        free_intlist(remove_samples)
        free_intlist(leaf_samples)

    cdef void retrain(self,
                      Node**    node_ptr,
                      DTYPE_t** X,
                      INT32_t*  y,
                      IntList*  remove_samples) nogil:
        """
        Rebuild subtree at this node.
        """
        # cdef Node** node_ptr = node_ptr_ptr[0]
        cdef Node*  node = node_ptr[0]

        # node properties
        cdef SIZE_t depth = node.depth
        cdef bint   is_left = node.is_left

        # printf('[R - R] copying constant_features\n')

        # get constant features array
        cdef IntList* constant_features = copy_intlist(node.constant_features, node.constant_features.n)

        # updated array of leaf samples
        cdef IntList* leaf_samples = create_intlist(node.n_samples, 0)
        get_leaf_samples(node, remove_samples, leaf_samples)

        # printf('[R - R] retrain, leaf_samples.n: %ld\n', leaf_samples.n)

        # free node / subtree
        dealloc(node)
        free(node)

        # printf('done freeing node\n')

        # retrain node / subtree
        node_ptr[0] = self.tree_builder._build(X, y, leaf_samples, constant_features, depth, is_left)

        # check complete: add deletion type and depth, and clean up
        # printf('[R - R] retrain, node_ptr[0].depth: %ld, node_ptr[0].n_features: %ld\n',
        #        node_ptr[0].depth, node_ptr[0].n_features)
        self.add_metric(1, node_ptr[0].depth, node_ptr[0].n_samples)
        free_intlist(remove_samples)

    cdef INT32_t check_optimal_split(self,
                                     Node* node) nogil:
        """
        Compare all feature thresholds against the chosen feature / threshold.
        """

        # parameters
        cdef bint   use_gini = self.config.use_gini
        cdef SIZE_t topd = self.config.topd

        # keep track of the best feature / threshold
        cdef DTYPE_t best_score = 1000000
        cdef DTYPE_t split_score = -1

        # record the best feature / threshold
        cdef SIZE_t  chosen_feature_ndx = 0
        cdef DTYPE_t chosen_threshold_value = 0

        # object pointers
        cdef Feature*   feature = NULL
        cdef Threshold* threshold = NULL

        # counters
        cdef SIZE_t j = 0
        cdef SIZE_t k = 0

        # return 1 if retraining is necessary, 0 otherwise
        cdef INT32_t result = 0

        # greedy node, check if the best feature / threshold has changed
        if node.depth >= topd:

            # printf('[R - COS] node.depth: %ld\n', node.depth)

            # retrain if the chosen feature / threshold is invalid
            if node.chosen_feature == NULL and node.chosen_threshold == NULL:
                return 1

            # find the best feature / threshold
            best_score = 1000000

            # printf('[R] node.n_features: %ld\n', node.n_features)

            # get thresholds for each feature
            for j in range(node.n_features):
                feature = node.features[j]
                # printf('[R] feature: %ld\n', feature.index)

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

                    # printf('[SN] feature.index: %ld, threshold.value: %.3f, split_score: %.3f\n',
                    #        feature.index, threshold.value, split_score)

                    # keep threshold with the best score
                    if split_score < best_score:
                        best_score = split_score
                        chosen_feature_ndx = feature.index
                        chosen_threshold_value = threshold.value

            # printf('[R - COS] node.chosen_feature.index: %ld, node.chosen_threshold.value: %.5f\n',
            #       node.chosen_feature.index, node.chosen_threshold.value)
            # printf('[R - COS] chosen_feature_ndx: %ld, chosen_threshold_value: %.5f\n',
            #       chosen_feature_ndx, chosen_threshold_value)

            # check to see if the same feature / threshold is still the best
            result = not (node.chosen_feature.index == chosen_feature_ndx and
                          node.chosen_threshold.value == chosen_threshold_value)

        # random node
        else:

            # valid threshold, no retraining
            result = 0

        return result


    cdef SIZE_t update_metadata(self,
                                Node*     node,
                                DTYPE_t** X,
                                INT32_t*  y,
                                IntList*  remove_samples) nogil:
        """
        Update each feature / threshold and return the number of usable thresholds.
        """

        # return variable
        cdef SIZE_t n_usable_thresholds = 0

        # greedy node
        if node.depth >= self.config.topd:
            n_usable_thresholds = self.update_greedy_node_metadata(node, X, y, remove_samples)

        # random node
        else:
            n_usable_thresholds = self.update_random_node_metadata(node, X, y, remove_samples)

        return n_usable_thresholds

    cdef SIZE_t update_greedy_node_metadata(self,
                                            Node*     node,
                                            DTYPE_t** X,
                                            INT32_t*  y,
                                            IntList*  remove_samples) nogil:
        """
        Update each threshold for all features at this node.
        """

        # class properties
        cdef SIZE_t k_samples = self.config.k
        cdef SIZE_t min_samples_leaf = self.config.min_samples_leaf
        cdef SIZE_t n_total_features = self.manager.n_features

        # counters
        cdef SIZE_t i = 0
        cdef SIZE_t j = 0
        cdef SIZE_t k = 0

        # object pointers
        cdef Feature*    feature = NULL
        cdef Threshold*  threshold = NULL

        # variables to keep track of invalid thresholds
        cdef DTYPE_t v1_label_ratio = -1
        cdef DTYPE_t v2_label_ratio = -1
        cdef SIZE_t* threshold_validities = NULL
        cdef SIZE_t  n_invalid_thresholds = 0
        cdef SIZE_t  n_valid_thresholds = 0
        cdef SIZE_t  n_new_thresholds = 0

        # invalid features
        cdef IntList* invalid_features = create_intlist(node.n_features, 0)

        # return variable
        cdef SIZE_t n_usable_thresholds = 0

        # printf('[R - UM] node.n_features: %ld\n', node.n_features)

        # update statistics for each feature
        for j in range(node.n_features):
            feature = node.features[j]

            # array holding indices of invalid thresholds for this feature
            threshold_validities = <SIZE_t *>malloc(feature.n_thresholds * sizeof(SIZE_t))

            # initialize counters
            n_invalid_thresholds = 0
            n_valid_thresholds = 0

            # printf('[R - UM] feature.index: %ld, feature.n_thresholds: %ld\n',
            #        feature.index, feature.n_thresholds)

            # update statistics for each threshold in this feature
            for k in range(feature.n_thresholds):
                threshold = feature.thresholds[k]

                # loop through each deleted sample
                for i in range(remove_samples.n):

                    # decrement left branch of this threshold
                    if X[remove_samples.arr[i]][feature.index] <= threshold.value:
                        threshold.n_left_samples -= 1
                        threshold.n_left_pos_samples -= y[remove_samples.arr[i]]

                    # decrement right branch of this threshold
                    else:
                        threshold.n_right_samples -= 1
                        threshold.n_right_pos_samples -= y[remove_samples.arr[i]]

                    # decrement left value of this threshold
                    if X[remove_samples.arr[i]][feature.index] == threshold.v1:
                        threshold.n_v1_samples -= 1
                        threshold.n_v1_pos_samples -= 1

                    # decrement right value of this threshold
                    elif X[remove_samples.arr[i]][feature.index] == threshold.v2:
                        threshold.n_v2_samples -= 1
                        threshold.n_v2_pos_samples -= 1

                # compute label ratios for both values of the threshold
                v1_label_ratio = threshold.n_v1_pos_samples / (1.0 * threshold.n_v1_samples)
                v2_label_ratio = threshold.n_v2_pos_samples / (1.0 * threshold.n_v2_samples)

                # check to see if threshold is still valid
                threshold_validities[k] = ((threshold.n_left_samples >= min_samples_leaf and
                                            threshold.n_right_samples >= min_samples_leaf) and
                                           (v1_label_ratio != v2_label_ratio or
                                           (v1_label_ratio > 0.0 and v2_label_ratio < 1.0)))

                # printf('[R - UM] threshold %.5f, n_left_samples: %ld, n_right_samples: %ld, valid: %ld\n',
                #        threshold.value, threshold.n_left_samples, threshold.n_right_samples, threshold_validities[k])

                # invalid threshold
                if not threshold_validities[k]:
                    n_invalid_thresholds += 1

                    # if the chosen feature / threshold is invalid, retrain
                    if (feature.index == node.chosen_feature.index and
                        threshold.value == node.chosen_threshold.value):

                        # printf('[R - UGNM] chosen feature / threshold has changed\n')

                        # clean up
                        free(threshold_validities)
                        free_intlist(invalid_features)
                        return -1

                # valid threshold
                else:
                    n_valid_thresholds += 1
                    n_usable_thresholds += 1

            # invalid thresholds, sample new viable thresholds, if any
            if n_invalid_thresholds > 0:
                # printf('[R - UGNM] n_invalid_thresholds: %ld\n', n_invalid_thresholds)

                # no other candidate thresholds for this feature
                if feature.n_thresholds < k_samples:

                    # remove invalid thresholds from this feature
                    if n_invalid_thresholds < feature.n_thresholds: 
                        # printf('[R - UGNM] no other candidates, remove invalid thresholds\n')
                        remove_invalid_thresholds(feature, n_valid_thresholds, threshold_validities)

                    # all thresholds invalid and no candidate thresholds, flag feature for replacement
                    else:
                        # printf('[R - UGNM] no other candidates, flag feature for replacement\n')
                        invalid_features.arr[invalid_features.n] = feature.index
                        invalid_features.n += 1

                # possibly other thresholds, sample new thresholds for this feature
                else:
                    # printf('[R - UGNM] possibly other thresholds, sample new thresholds\n')
                    n_new_thresholds = sample_new_thresholds(feature, n_valid_thresholds,
                                                             threshold_validities, node, X, y,
                                                             remove_samples, NULL, self.config)

                    # all thresholds invalid, flag feature for replacement
                    if n_new_thresholds == 0 and n_valid_thresholds == 0:
                        # printf('[R - UGNM] all thresholds invalid, flag feature for replacement\n')
                        invalid_features.arr[invalid_features.n] = feature.index
                        invalid_features.n += 1

                    # increment no. usable thresholds
                    n_usable_thresholds += n_new_thresholds

            # clean up
            free(threshold_validities)

        # replace invalid features
        # printf('[R - UGNM] invalid_features.n: %ld\n', invalid_features.n)
        if invalid_features.n > 0:
            n_usable_thresholds += sample_new_features(&node.features, &node.constant_features,
                                                       invalid_features, n_total_features, node, X, y,
                                                       remove_samples, self.config)

            # printf('[R - UGNM] node.n_features: %ld\n', node.n_features)

            # no usable thresholds if there are no valid features
            if node.n_features == 0:
                n_usable_thresholds = 0

        # clean up
        free_intlist(invalid_features)

        return n_usable_thresholds

    cdef SIZE_t update_random_node_metadata(self,
                                            Node*     node,
                                            DTYPE_t** X,
                                            INT32_t*  y,
                                            IntList*  remove_samples) nogil:
        """
        Update the metadata for the chosen feature / threshold of the
        random node.
        """

        # configuration
        cdef SIZE_t min_samples_leaf = self.config.min_samples_leaf

        # return variable
        cdef SIZE_t n_usable_thresholds = 1

        # loop through samples to be removed
        for i in range(remove_samples.n):

            # decrement left branch sample count
            if X[remove_samples.arr[i]][node.chosen_feature.index] <= node.chosen_threshold.value:
                node.chosen_threshold.n_left_samples -= 1

            # decrement right branch sample count
            else:
                node.chosen_threshold.n_right_samples -= 1

        # not enough samples in both branches, invalid threshold
        if (node.chosen_threshold.n_left_samples < min_samples_leaf or
            node.chosen_threshold.n_right_samples < min_samples_leaf):

            # retrain
            n_usable_thresholds = -1

        return n_usable_thresholds

    cdef void add_metric(self,
                         INT32_t remove_type,
                         INT32_t remove_depth,
                         INT32_t remove_cost) nogil:
        """
        Add type and depth to the removal metrics.
        """
        if self.remove_types:
            if self.remove_count + 1 == self.capacity:
                self.capacity *= 2

            self.remove_types = <INT32_t *>realloc(self.remove_types, self.capacity * sizeof(INT32_t))
            self.remove_depths = <INT32_t *>realloc(self.remove_depths, self.capacity * sizeof(INT32_t))
            self.remove_costs = <INT32_t *>realloc(self.remove_costs, self.capacity * sizeof(INT32_t))

        else:
            self.capacity = 10
            self.remove_types = <INT32_t *>malloc(self.capacity * sizeof(INT32_t))
            self.remove_depths = <INT32_t *>malloc(self.capacity * sizeof(INT32_t))
            self.remove_costs = <INT32_t *>malloc(self.capacity * sizeof(INT32_t))

        self.remove_types[self.remove_count] = remove_type
        self.remove_depths[self.remove_count] = remove_depth
        self.remove_costs[self.remove_count] = remove_cost
        self.remove_count += 1

    cpdef void clear_metrics(self):
        """
        Resets deletion statistics.
        """
        free(self.remove_types)
        free(self.remove_depths)
        free(self.remove_costs)
        self.remove_count = 0
        self.remove_types = NULL
        self.remove_depths = NULL
        self.remove_costs = NULL

    cdef np.ndarray get_int_ndarray(self,
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


# helper methods
cdef void remove_invalid_thresholds(Feature* feature,
                                    SIZE_t   n_valid_thresholds,
                                    SIZE_t*  threshold_validities) nogil:
    """
    Removes the invalid thresholds from this feature.
    """

    # create final thresholds array
    cdef Threshold** new_thresholds = <Threshold **>malloc(n_valid_thresholds  * sizeof(Threshold *))
    cdef SIZE_t      n_new_thresholds = 0

    # counters
    cdef SIZE_t k = 0

    # filter out invalid thresholds and free original thresholds array
    for k in range(feature.n_thresholds):

        # add valid threshold to final thresholds
        if threshold_validities[k] == 1:
            new_thresholds[n_new_thresholds] = copy_threshold(feature.thresholds[k])
            n_new_thresholds += 1

    # free thresholds array
    free_thresholds(feature.thresholds, feature.n_thresholds)

    # save new thresholds array to the feature
    feature.thresholds = new_thresholds
    feature.n_thresholds = n_new_thresholds


cdef SIZE_t sample_new_thresholds(Feature*  feature,
                                  SIZE_t    n_valid_thresholds,
                                  SIZE_t*   threshold_validities,
                                  Node*     node,
                                  DTYPE_t** X,
                                  INT32_t*  y,
                                  IntList*  remove_samples,
                                  bint*     is_constant_feature_ptr,
                                  _Config   config) nogil:
    """
    Try to sample new thresholds for all invalid thresholds for this feature.
    """

    # configuration
    cdef SIZE_t    k_samples = config.k
    cdef SIZE_t    min_samples_leaf = config.min_samples_leaf
    cdef UINT32_t* random_state = &config.rand_r_state

    # iterators
    cdef SIZE_t  i = 0
    cdef SIZE_t  k = 0

    # samplers
    cdef INT32_t ndx = 0

    # candidate threshold variables
    cdef Threshold** candidate_thresholds = NULL
    cdef SIZE_t      n_candidate_thresholds = 0

    # unused candidate threshold array variables
    cdef Threshold** unused_thresholds = NULL
    cdef SIZE_t      n_unused_thresholds = 0
    cdef SIZE_t      n_unused_thresholds_to_sample = 0
    cdef SIZE_t      n_vacancies = 0

    # variables for tracking
    cdef IntList*    sampled_indices = NULL
    cdef bint        valid = True

    # result variables
    cdef Threshold** new_thresholds = NULL
    cdef SIZE_t      n_new_thresholds = 0

    # return variable
    cdef SIZE_t n_usable_thresholds = 0

    # get updated samples for this node
    cdef IntList* samples = create_intlist(node.n_samples, 0)
    get_leaf_samples(node, remove_samples, samples)

    # helper variables
    cdef DTYPE_t* values = <DTYPE_t *>malloc(samples.n * sizeof(DTYPE_t))
    cdef INT32_t* labels = <INT32_t *>malloc(samples.n * sizeof(INT32_t))
    cdef SIZE_t*  indices = <SIZE_t *>malloc(samples.n * sizeof(SIZE_t))
    cdef SIZE_t   n_pos_samples = 0

    # copy values and labels into new arrays, and count no. pos. labels
    for i in range(samples.n):
        values[i] = X[samples.arr[i]][feature.index]
        labels[i] = y[samples.arr[i]]
        indices[i] = i
        n_pos_samples += y[samples.arr[i]]

    # sort feature values, and their corresponding indices
    sort(values, indices, samples.n)

    # constant feature
    if fabs(values[samples.n - 1] - values[0]) <= FEATURE_THRESHOLD:
        if is_constant_feature_ptr != NULL:
            is_constant_feature_ptr[0] = True

    # get candidate thresholds
    candidate_thresholds = <Threshold **>malloc(samples.n * sizeof(Threshold *))
    n_candidate_thresholds = get_candidate_thresholds(values, labels, indices,
                                                      samples.n, n_pos_samples,
                                                      min_samples_leaf, &candidate_thresholds)

    # array of candidate thresholds that are not being used
    unused_thresholds = <Threshold **>malloc(n_candidate_thresholds * sizeof(Threshold *))
    n_unused_thresholds = 0

    # filter out already used thresholds
    for i in range(n_candidate_thresholds):
        used = False

        # disregard threshold if it is already being used by the feature
        for k in range(feature.n_thresholds):
            if feature.thresholds[k].value == candidate_thresholds[i].value:
                used = True
                break

        # add candidate threshold to list of unused candidate thresholds
        if not used:
            unused_thresholds[n_unused_thresholds] = candidate_thresholds[i]
            n_unused_thresholds += 1

    # compute number of unused thresholds to sample
    n_vacancies = k_samples - n_valid_thresholds
    if n_unused_thresholds > n_vacancies:
        n_unused_thresholds_to_sample = n_vacancies
    else:
        n_unused_thresholds_to_sample = n_unused_thresholds

    # allocate memory for the new thresholds array
    n_new_thresholds = n_unused_thresholds_to_sample + n_valid_thresholds
    new_thresholds = <Threshold **>malloc(n_new_thresholds * sizeof(Threshold *))
    sampled_indices = create_intlist(n_unused_thresholds_to_sample, 0)

    # sample unused threshold indices uniformly at random
    i = 0
    while sampled_indices.n < n_unused_thresholds_to_sample:
        valid = True

        # sample an unused threshold index
        ndx = rand_int(0, n_unused_thresholds, random_state)

        # invalid: already sampled
        for k in range(sampled_indices.n):
            if ndx == sampled_indices.arr[k]:
                valid = False
                break

        # valid: copy threshold to new thresholds
        if valid:
            new_thresholds[sampled_indices.n] = copy_threshold(unused_thresholds[ndx])
            sampled_indices.arr[sampled_indices.n] = ndx
            sampled_indices.n += 1
            i += 1

    # add original thresholds to the new thresholds array
    for k in range(feature.n_thresholds):

        # original valid threshold
        if threshold_validities[k] == 1:
            new_thresholds[i] = copy_threshold(feature.thresholds[k])
            i += 1

    # free previous thresholds and candidate thresholds arrays
    free_thresholds(feature.thresholds, feature.n_thresholds)
    free_thresholds(candidate_thresholds, n_candidate_thresholds)

    # clean up
    free(values)
    free(labels)
    free(indices)
    free(unused_thresholds)
    free_intlist(samples)
    free_intlist(sampled_indices)

    # set new thresholds array for this feature
    feature.thresholds = new_thresholds
    feature.n_thresholds = n_new_thresholds

    return n_unused_thresholds_to_sample


cdef SIZE_t sample_new_features(Feature*** features_ptr,
                                IntList**  constant_features_ptr,
                                IntList*   invalid_features,
                                SIZE_t     n_total_features,
                                Node*      node,
                                DTYPE_t**  X,
                                INT32_t*   y,
                                IntList*   remove_samples,
                                _Config    config) nogil:
    """
    Sample new unused features to replace the invalid features.
    """

    # configuration
    cdef SIZE_t max_features = config.max_features
    cdef UINT32_t* random_state = &config.rand_r_state

    # indexers
    cdef SIZE_t feature_index = 0
    cdef SIZE_t invalid_feature_index = 0

    # counters
    cdef SIZE_t i = 0
    cdef SIZE_t j = 0
    cdef SIZE_t k = 0
    cdef SIZE_t n_used_invalid_features = 0

    # keeps track of invalid features
    cdef IntList* constant_features = copy_intlist(constant_features_ptr[0], n_total_features)
    cdef IntList* sampled_features = create_intlist(n_total_features, 0)
    cdef SIZE_t   n_features = node.n_features - invalid_features.n
    cdef bint     is_constant_feature = False
    cdef bint     valid = False

    # old and new features array
    cdef Feature** features = features_ptr[0]
    cdef Feature** new_features = NULL
    cdef SIZE_t    n_new_features = 0

    # return variable
    cdef SIZE_t n_usable_thresholds = 0

    # printf('[R - SNF] node.n_features: %ld, invalid_features.n: %ld\n', node.n_features, invalid_features.n)

    # add valid and invalid features to the list of sampled features
    for j in range(node.n_features):
        sampled_features.arr[sampled_features.n] = node.features[j].index
        sampled_features.n += 1

    # sample features until the previous no. features is reached or there are no features left
    while n_features < node.n_features and (sampled_features.n + constant_features.n) < n_total_features:

        # sample feature index
        feature_index = rand_int(0, n_total_features, random_state)

        # already sampled feature
        valid = True
        for i in range(sampled_features.n):
            if sampled_features.arr[i] == feature_index:
                valid = False
                break
        if not valid:
            continue

        # known constant feature
        valid = True
        for i in range(constant_features.n):
            if constant_features.arr[i] == feature_index:
                valid = False
                break
        if not valid:
            continue

        # create a feature and sample thresholds
        feature = create_feature(feature_index)
        sample_new_thresholds(feature, 0, NULL, node, X, y, remove_samples, &is_constant_feature, config)

        # printf('[R - SNF] is_constant_feature: %d\n', is_constant_feature)

        # constant feature
        if is_constant_feature:
            constant_features.arr[constant_features.n] = feature_index
            constant_features.n += 1
            free_feature(feature)
            continue

        # add to sampled features array
        sampled_features.arr[sampled_features.n] = feature_index
        sampled_features.n += 1

        # no valid thresholds, free feature and sample a new feature
        if feature.n_thresholds == 0:
            # printf('[R - SNF] no valid thresholds, free_feature\n')
            free_feature(feature)
            continue

        # printf('[R - SNF] feature.n_thresholds: %ld\n', feature.n_thresholds)

        # get invalid feature index
        invalid_feature_index = invalid_features.arr[n_used_invalid_features]
        n_used_invalid_features += 1

        # replace the invalid feature with a valid one
        for j in range(node.n_features):
            if features[j].index == invalid_feature_index:
                free_feature(features[j])
                features[j] = feature
                n_features += 1
                n_usable_thresholds += feature.n_thresholds
                break

    # could not replace all invalid features, remove remaining invalid features
    if n_features < node.n_features:

        # printf('[R - SNF] n_features: %ld, node.n_features: %ld\n', n_features, node.n_features)

        # new (smaller) features array
        new_features = <Feature **>malloc(n_features * sizeof(Feature *))
        n_new_features = 0

        # copy valid features to this new array
        for j in range(node.n_features):
            valid = True

            # invalid feature
            for k in range(invalid_features.n):
                if features[j].index == invalid_features.arr[k]:
                    valid = False
                    break

            # copy valid feature to new features array
            if valid:
                new_features[n_new_features] = copy_feature(features[j])
                n_new_features += 1

        # free previous features
        free_features(features, node.n_features)

        # set features to new features array
        features_ptr[0] = new_features

    # printf('[R - SNF] freeing constant_features\n')

    # free previous constant features array and set new constant features array
    free_intlist(constant_features_ptr[0])
    constant_features_ptr[0] = copy_intlist(constant_features, constant_features.n)

    # set no. features
    node.n_features = n_features

    # clean up
    free_intlist(sampled_features)
    free_intlist(constant_features)

    return n_usable_thresholds


cdef void get_leaf_samples(Node*    node,
                           IntList* remove_samples,
                           IntList* leaf_samples) nogil:
    """
    Recursively obtain the samples at the leaves and filter out
    deleted samples.
    """
    cdef bint   add_sample = True
    cdef SIZE_t i = 0
    cdef SIZE_t j = 0

    # leaf
    if node.is_leaf:

        # loop through all samples at this leaf
        for i in range(node.n_samples):
            add_sample = True

            # loop through all deleted samples
            for j in range(remove_samples.n):

                # printf('[R - GLS] node.leaf_samples[%ld]: %ld, remove_samples[%ld]: %ld\n',
                #        i, node.leaf_samples[i], j, remove_samples[j])

                # do not add sample to results if it has been deleted
                if node.leaf_samples[i] == remove_samples.arr[j]:
                    add_sample = False
                    break

            # add sample to results if it has not been deleted
            if add_sample:
                leaf_samples.arr[leaf_samples.n] = node.leaf_samples[i]
                leaf_samples.n += 1

    # decision node
    else:

        # traverse left
        if node.left:
            get_leaf_samples(node.left, remove_samples, leaf_samples)

        # traverse right
        if node.right:
            get_leaf_samples(node.right, remove_samples, leaf_samples)
