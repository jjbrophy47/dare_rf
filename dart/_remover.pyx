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
from ._splitter cimport get_candidate_thresholds
from ._utils cimport compute_split_score
from ._utils cimport rand_uniform
from ._utils cimport split_samples
from ._utils cimport copy_threshold
from ._utils cimport convert_int_ndarray
from ._utils cimport copy_int_array
from ._utils cimport dealloc
from ._utils cimport RAND_R_MAX

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

    property retrain_sample_count:
        def __get__(self):
            return self.retrain_sample_count

    def __cinit__(self,
                  _DataManager manager,
                  _TreeBuilder tree_builder,
                  bint         use_gini,
                  SIZE_t       k,
                  object       random_state):
        """
        Constructor.
        """
        self.manager = manager
        self.tree_builder = tree_builder
        self.use_gini = use_gini
        self.k = k
        self.rand_r_state = random_state.randint(0, RAND_R_MAX)

        # initialize metric properties
        self.capacity = 10
        self.remove_count = 0
        self.remove_types = <INT32_t *>malloc(self.capacity * sizeof(INT32_t))
        self.remove_depths = <INT32_t *>malloc(self.capacity * sizeof(INT32_t))

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
        cdef SIZE_t n_usable_thresholds = 0

        printf('\n[R] n_samples: %ld, depth: %ld, is_left: %d\n', n_samples, node.depth, node.is_left)

        # update node counts
        n_pos_samples = self.update_node(&node, y, samples, n_samples)

        # leaf, check complete
        if node.is_leaf:
            printf('[R] update leaf\n')
            self.update_leaf(&node, samples, n_samples)
            printf('[R] leaf.value: %.2f\n', node.value)

        # decision node, but samples in same class, convert to leaf, check complete
        elif node.n_pos_samples == 0 or node.n_pos_samples == node.n_samples:
            printf('[R] convert to leaf\n')
            self.convert_to_leaf(&node, samples, n_samples, &split)
            printf('[R] convert to leaf, leaf.value: %.2f\n', node.value)

        # decision node
        else:

            # update metadata
            printf('[R] update metadata\n')
            n_usable_thresholds = self.update_metadata(&node, X, y, samples, n_samples)
            printf('[R] n_usable_thresholds: %ld\n', n_usable_thresholds)

            # no more usable thresholds, convert to leaf, check complete
            if n_usable_thresholds == 0:
                printf('[R] convert to leaf\n')
                self.convert_to_leaf(&node, samples, n_samples, &split)
                printf('[R] convert to leaf, leaf.value: %.2f\n', node.value)

            # viable decision node
            else:

                # check optimal split
                printf('[R] check optimal split\n')
                result = self.check_optimal_split(node)

                # optimal split has changed, retrain, check complete
                if result == 1:
                    printf('[R] retrain\n')
                    self.retrain(&node_ptr, X, y, samples, n_samples)
                    printf('[R] retrain, n_samples: %.ld\n', node.n_samples)

                # no retraining necessary, split samples and recurse
                else:

                    # split deleted samples and free original samples
                    split_samples(node, X, y, samples, n_samples, &split)
                    printf('[R] split samples\n')

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
        node.n_samples -= n_samples
        node.n_pos_samples -= n_pos_samples

        # return no. positive samples being deleted
        return n_pos_samples

    cdef void update_leaf(self,
                          Node**  node_ptr,
                          SIZE_t* remove_samples,
                          SIZE_t  n_remove_samples) nogil:
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
        node.n_samples += n_remove_samples  # must check ALL leaf samples, even deleted ones
        self.get_leaf_samples(node, remove_samples, n_remove_samples, &leaf_samples, &leaf_samples_count)
        node.n_samples -= n_remove_samples

        # free old leaf samples array
        free(node.leaf_samples)

        # assign new leaf samples and value
        node.leaf_samples = leaf_samples

        # check complete: add deletion type and depth, and clean up
        self.add_removal_type(0, node.depth)
        free(remove_samples)

    cdef void convert_to_leaf(self,
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
        cdef bint   add_sample = True
        cdef SIZE_t i = 0
        cdef SIZE_t j = 0

        # leaf
        if node.is_leaf:

            # loop through all samples at this leaf
            for i in range(node.n_samples):
                add_sample = True

                # loop through all deleted samples
                for j in range(n_remove_samples):
                    # printf('[R - GLS] samples[%ld]: %ld, remove_samples[%ld]: %ld\n',
                           # i, node.leaf_samples[i], j, remove_samples[j])

                    # do not add sample to results if it has been deleted
                    if node.leaf_samples[i] == remove_samples[j]:
                        # printf('[R - GLS] DO NOT ADD!\n')
                        add_sample = False
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

    cdef INT32_t check_optimal_split(self,
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
        cdef SIZE_t  chosen_feature_ndx = 0
        cdef DTYPE_t chosen_threshold_value = 0

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

            # retrain if the chosen feature / threshold is invalid
            if node.chosen_feature == NULL and node.chosen_threshold == NULL:
                return 1

            # find the best feature / threshold
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

            # printf('[R - COS] node.chosen_feature.index: %ld, node.chosen_threshold.value: %.5f\n',
            #       node.chosen_feature.index, node.chosen_threshold.value)
            # printf('[R - COS] chosen_feature_ndx: %ld, chosen_threshold_value: %.5f\n',
            #       chosen_feature_ndx, chosen_threshold_value)

            # check to see if the same feature / threshold is still the best
            result = not (node.chosen_feature.index == chosen_feature_ndx and
                          node.chosen_threshold.value == chosen_threshold_value)

            # printf('[R - COS] result: %d\n', result)

        return result

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

        # class properties
        cdef SIZE_t    k_samples = self.k
        cdef UINT32_t* random_state = &self.rand_r_state

        # counters
        cdef int i = 0
        cdef int j = 0
        cdef int k = 0

        # object pointers
        cdef Feature*    feature = NULL
        cdef Threshold*  threshold = NULL

        # variables to keep track of invalid thresholds
        cdef DTYPE_t v1_label_ratio = -1
        cdef DTYPE_t v2_label_ratio = -1
        cdef SIZE_t* threshold_validities = NULL
        cdef SIZE_t  n_invalid_thresholds = 0
        cdef SIZE_t  n_valid_thresholds = 0
        cdef bint    valid_threshold = False

        # leaf samples array
        cdef SIZE_t* leaf_samples = NULL
        cdef SIZE_t  n_leaf_samples = 0

        # candidate thresholds array
        cdef Threshold** candidate_thresholds = NULL
        cdef SIZE_t      n_candidate_thresholds = 0

        # unused candidate thresholds array
        cdef bint        used = False
        cdef Threshold** unused_thresholds = NULL
        cdef SIZE_t      n_unused_thresholds = 0

        # variables for sampling the unused candidate array
        cdef SIZE_t      n_vacancies = 0
        cdef SIZE_t      n_thresholds_to_sample = 0
        cdef Threshold** final_thresholds = NULL
        cdef SIZE_t      n_final_thresholds = 0
        cdef INT32_t     ndx = -1
        cdef SIZE_t*     sampled_indices = NULL

        # return variable
        cdef SIZE_t n_usable_thresholds = 0

        # update statistics for each feature
        for j in range(node.n_features):
            feature = node.features[j]

            # array holding indices of invalid thresholds for this feature
            threshold_validities = <SIZE_t *>malloc(feature.n_thresholds * sizeof(SIZE_t))

            # initialize counters
            n_invalid_thresholds = 0
            n_valid_thresholds = 0

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

                # compute label ratios for both values of the threshold
                v1_label_ratio = threshold.n_v1_pos_samples / (1.0 * threshold.n_v1_samples)
                v2_label_ratio = threshold.n_v2_pos_samples / (1.0 * threshold.n_v2_samples)

                # check to see if threshold is still valid
                valid_threshold = ((threshold.n_left_samples > 0 and threshold.n_right_samples > 0) and
                                  ((v1_label_ratio != v2_label_ratio) or
                                   (v1_label_ratio > 0.0 and v1_label_ratio < 1.0) or
                                   (v2_label_ratio > 0.0 and v2_label_ratio < 1.0)))

                # save whether this threshold was valid or not
                threshold_validities[k] = valid_threshold

                # invalid threshold, flag for removal
                if not valid_threshold:
                    n_invalid_thresholds += 1

                    # if the chosen feature / threshold is invalid, clear those properties
                    if ((feature.index == node.chosen_feature.index) and
                        (threshold.value == node.chosen_threshold.value)):
                        node.chosen_feature = NULL
                        node.chosen_threshold = NULL

                # valid threshold
                else:
                    n_valid_thresholds += 1
                    n_usable_thresholds += 1

            printf('[R - UM] feature.index: %ld, n_valid_thresholds: %ld, n_invalid_thresholds: %ld\n',
                   feature.index, n_valid_thresholds, n_invalid_thresholds)

            # sample new viable thresholds, if any
            if n_invalid_thresholds > 0:
                printf('[R - UM] sample new thresholds, feauture.n_thresholds: %ld, k_samples: %ld\n',
                       feature.n_thresholds, k_samples)

                # if no. original thresholds < k, there are no other candidate thresholds
                if feature.n_thresholds == k_samples:

                    # get instances for this node, filter out deleted instances
                    leaf_samples = <SIZE_t *>malloc(node.n_samples * sizeof(SIZE_t))
                    n_leaf_samples = 0
                    self.get_leaf_samples(node, samples, n_samples, &leaf_samples, &n_leaf_samples)

                    printf('[R - UM] n_leaf_samples: %ld\n', n_leaf_samples)

                    # extract candidate thresholds
                    candidate_thresholds = <Threshold **>malloc(n_leaf_samples * sizeof(Threshold *))
                    n_candidate_thresholds = get_candidate_thresholds(feature, X, y, leaf_samples,
                                                                      n_leaf_samples, &candidate_thresholds)
                    printf('[R - UM] n_candidate_thresholds: %ld\n', n_candidate_thresholds)

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
                        n_thresholds_to_sample = n_vacancies
                    else:
                        n_thresholds_to_sample = n_unused_thresholds
                    n_usable_thresholds += n_thresholds_to_sample

                    # allocate memory for the new thresholds array
                    n_final_thresholds = n_thresholds_to_sample + n_valid_thresholds
                    final_thresholds = <Threshold **>malloc(n_final_thresholds * sizeof(Threshold *))
                    sampled_indices = <SIZE_t *>malloc(n_thresholds_to_sample * sizeof(SIZE_t))

                    # sample unused threshold indices uniformly at random
                    i = 0
                    while i < n_thresholds_to_sample:
                        valid = True

                        # sample an unused threshold index
                        ndx = <INT32_t>(rand_uniform(0, 1, random_state) / (1.0 / n_unused_thresholds))

                        # invalid: already sampled
                        for k in range(i):
                            if ndx == sampled_indices[k]:
                                valid = False
                                break

                        # valid: add copied threshold to sampled list of candidate thresholds
                        if valid:
                            final_thresholds[i] = copy_threshold(unused_thresholds[ndx])
                            sampled_indices[i] = ndx
                            i += 1

                    # add original thresholds to the new thresholds array
                    for k in range(feature.n_thresholds):

                        # original valid threshold
                        if threshold_validities[k] == 1:
                            final_thresholds[i] = copy_threshold(feature.thresholds[k])
                            i += 1

                        # free threshold
                        free(feature.thresholds[k])
                    free(feature.thresholds)

                    # free candidate thresholds array
                    for k in range(n_candidate_thresholds):
                        free(candidate_thresholds[k])
                    free(candidate_thresholds)

                    # clean up
                    free(unused_thresholds)
                    free(leaf_samples)
                    free(sampled_indices)

                # remove invalid thresholds
                else:

                    printf('[R - UM] remove invalid thresholds\n')

                    # create final thresholds array
                    n_final_thresholds = feature.n_thresholds - n_invalid_thresholds
                    final_thresholds = <Threshold **>malloc(n_final_thresholds  * sizeof(Threshold *))

                    # filter out invalid thresholds and free original thresholds array
                    i = 0
                    for k in range(feature.n_thresholds):

                        # add valid threshold to final thresholds
                        if threshold_validities[k] == 1:
                            final_thresholds[i] = copy_threshold(feature.thresholds[k])
                            i +=1

                        free(feature.thresholds[k])
                    free(feature.thresholds)

                # assign final thresholds array to this feature
                feature.thresholds = final_thresholds
                feature.n_thresholds = n_final_thresholds

            # clean up
            free(threshold_validities)

        return n_usable_thresholds

    cdef void add_removal_type(self,
                               INT32_t remove_type,
                               INT32_t remove_depth) nogil:
        """
        Add type and depth to the removal metrics.
        """
        if self.remove_types and self.remove_depths:
            if self.remove_count + 1 == self.capacity:
                self.capacity *= 2

            self.remove_types = <INT32_t *>realloc(self.remove_types, self.capacity * sizeof(INT32_t))
            self.remove_depths = <INT32_t *>realloc(self.remove_depths, self.capacity * sizeof(INT32_t))

        else:
            self.capacity = 10
            self.remove_types = <INT32_t *>malloc(self.capacity * sizeof(INT32_t))
            self.remove_depths = <INT32_t *>malloc(self.capacity * sizeof(INT32_t))

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
