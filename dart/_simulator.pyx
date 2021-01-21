# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
"""
Data deletion simulator.
"""
from cpython cimport Py_INCREF
from cpython.ref cimport PyObject

cimport cython

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf

import numpy as np
cimport numpy as np
np.import_array()

from ._tree cimport Feature
from ._tree cimport Threshold
from ._splitter cimport get_candidate_thresholds
from ._utils cimport compute_split_score
from ._utils cimport rand_uniform
from ._utils cimport copy_threshold
from ._utils cimport copy_feature

# =====================================
# Remover
# =====================================

cdef class _Simulator:
    """
    Class to simulate deletion operations without actually
    updating the model and database.
    """

    def __cinit__(self,
                  _DataManager manager,
                  _Config      config):
        """
        Constructor.
        """
        self.manager = manager
        self.config = config

    cpdef INT32_t sim_delete(self, _Tree tree, SIZE_t remove_index):
        """
        Simualate the deletion of a training sample.

        Returns the number of samples needed for retraining due to the
        deletion of the `remove_index`.
        """

        # Data containers
        cdef DTYPE_t** X = NULL
        cdef INT32_t*  y = NULL
        self.manager.get_data(&X, &y)

        # check if any sample has already been deleted
        cdef INT32_t result = self.manager.check_single_remove_sample_validity(remove_index)
        if result == -1:
            return -1

        # save random state
        cdef UINT32_t prev_rand_r_state = self.config.rand_r_state

        # recurse through the tree and return the number of samples needed for retraining
        result = self._sim_delete(&tree.root, X, y, remove_index)

        # restore random state before the simulation
        self.config.rand_r_state = prev_rand_r_state

        return result

    cdef INT32_t _sim_delete(self,
                             Node**    node_ptr,
                             DTYPE_t** X,
                             INT32_t*  y,
                             SIZE_t    remove_index) nogil:
        """
        Traverse tree until a stopping criterion is reached.

        DO NOT MODIFY ANY NODE PROPERTIES.
        """
        cdef Node *node = node_ptr[0]

        # get updated node statistics
        cdef SIZE_t node_n_samples = node.n_samples - 1
        cdef SIZE_t node_n_pos_samples = node.n_pos_samples - y[remove_index]

        # features array to hold updated feature / threshold statistics
        cdef Feature** features = NULL
        cdef SIZE_t    n_features = 0

        # counters
        cdef SIZE_t j = 0

        # result container
        cdef INT32_t result = 0

        # printf('\n[S] node_n_samples: %ld, depth: %ld, is_left: %d\n', node_n_samples, node.depth, node.is_left)

        # leaf, no retraining
        if node.is_leaf:
            # printf('[S] leaf.value: %.2f\n', node.value)
            return 0

        # decision node, but all samples in same class, convert to leaf, no retraining
        elif node_n_pos_samples == 0 or node_n_pos_samples == node_n_samples:
            # printf('[S] all samples in same class, convert to leaf\n')
            return 0

        # decision node, check if optimal threshold has changed
        else:

            # copy features array
            features = <Feature **>malloc(node.n_features * sizeof(Feature *))
            n_features = node.n_features
            for j in range(n_features):
                features[j] = copy_feature(node.features[j])

            # update features array
            n_valid_thresholds = self.update_metadata(node, X, y, remove_index, features, n_features)

            # no valid thresholds, convert to leaf, no retraining
            if n_valid_thresholds == 0:
                # printf('[S] no valid thresholds, convert to leaf\n')
                return 0

            # 1+ valid thresholds, but the chosen feature / threshold is invalid, retraining
            elif n_valid_thresholds < 0:
                # printf('[S] chosen feature / threshold is invalid, retrain %ld samples\n', node_n_samples)
                return node_n_samples

            result = self.check_optimal_split(node, features, n_features)

            # retraining necessary, return no. samples at this node
            if result == 1:
                # printf('[S] new optimal feature / threshold, depth: %ld, retrain %ld samples\n',
                #        node.depth, node_n_samples)
                return node_n_samples

            # optimal feature / threshold has not changed, move to next node
            else:

                # traverse left if deleted sample goes left
                if X[remove_index][node.chosen_feature.index] <= node.chosen_threshold.value:
                    return self._sim_delete(&node.left, X, y, remove_index)

                # traverse right if deleted sample goes right
                else:
                    return self._sim_delete(&node.right, X, y, remove_index)

    # private
    cdef INT32_t check_optimal_split(self,
                                     Node*     node,
                                     Feature** features,
                                     SIZE_t    n_features) nogil:
        """
        Compare all feature thresholds against the chosen feature / threshold.

        DO NOT MODIFY ANY NODE PROPERTIES.
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
        cdef SIZE_t i = 0
        cdef SIZE_t j = 0
        cdef SIZE_t k = 0

        # return 1 if retraining is necessary, 0 otherwise
        cdef INT32_t result = 0

        # greedy node, check if the best feature / threshold has changed
        if node.depth >= topd:

            # find the best feature / threshold
            best_score = 1000000

            # get thresholds for each feature
            for j in range(n_features):
                feature = features[j]

                # compute split score for each threshold
                for k in range(feature.n_thresholds):
                    threshold = feature.thresholds[k]

                    # compute split score, entropy or Gini index
                    split_score = compute_split_score(use_gini,
                                                      node.n_samples - 1,
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

            # printf('[R - COS] result: %d\n', result)

        return result

    cdef SIZE_t update_metadata(self,
                                Node*      node,
                                DTYPE_t**  X,
                                INT32_t*   y,
                                SIZE_t     remove_index,
                                Feature**  features,
                                SIZE_t     n_features) nogil:
        """
        Update each threshold for all features at this node.

        DO NOT MODIFY ANY NODE PROPERTIES.
        """

        # class properties
        cdef SIZE_t    k_samples = self.config.k
        cdef UINT32_t* random_state = &self.config.rand_r_state

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
        cdef bint    valid_threshold = False
        cdef bint    chosen_feature_threshold_invalid = False

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
        for j in range(n_features):
            feature = features[j]

            # printf('[R - UM] feature.index: %ld, feature.n_thresholds: %ld\n',
            #        feature.index, feature.n_thresholds)

            # array holding indices of invalid thresholds for this feature
            threshold_validities = <SIZE_t *>malloc(feature.n_thresholds * sizeof(SIZE_t))

            # initialize counters
            n_invalid_thresholds = 0
            n_valid_thresholds = 0

            # update statistics for each threshold in this feature
            for k in range(feature.n_thresholds):
                threshold = feature.thresholds[k]

                # decrement left branch of this threshold
                if X[remove_index][feature.index] <= threshold.value:
                    threshold.n_left_samples -= 1
                    threshold.n_left_pos_samples -= y[remove_index]

                # decrement right branch of this threshold
                else:
                    threshold.n_right_samples -= 1
                    threshold.n_right_pos_samples -= y[remove_index]

                # decrement left value of this threshold
                if X[remove_index][feature.index] == threshold.v1:
                    threshold.n_v1_samples -= 1
                    threshold.n_v1_pos_samples -= 1

                # decrement right value of this threshold
                elif X[remove_index][feature.index] == threshold.v2:
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

                # printf('[S - UM] threshold %.5f, n_left_samples: %ld, n_right_samples: %ld, valid: %ld\n',
                #        threshold.value, threshold.n_left_samples, threshold.n_right_samples, threshold_validities[k])

                # invalid threshold
                if not valid_threshold:
                    n_invalid_thresholds += 1

                    # chosen feature / threshold is invalid
                    if (feature.index == node.chosen_feature.index and
                        threshold.value == node.chosen_threshold.value):
                        chosen_feature_threhsold_invalid = True

                # valid threshold
                else:
                    n_valid_thresholds += 1
                    n_usable_thresholds += 1

            # chosen feature / threshold is invalid and there are usable thresholds, retrain
            if chosen_feature_threshold_invalid and n_usable_thresholds > 0:
                free(threshold_validities)
                return -1

            # sample new viable thresholds, if any
            if n_invalid_thresholds > 0:

                # if no. original thresholds < k, there are no other candidate thresholds
                if feature.n_thresholds == k_samples:

                    # printf('[S - UM] node.n_samples: %ld\n', node.n_samples - 1)

                    # get instances for this node, filter out deleted instances
                    leaf_samples = <SIZE_t *>malloc((node.n_samples - 1) * sizeof(SIZE_t))
                    n_leaf_samples = 0
                    self.get_leaf_samples(node, remove_index, &leaf_samples, &n_leaf_samples)

                    # extract candidate thresholds
                    candidate_thresholds = <Threshold **>malloc(n_leaf_samples * sizeof(Threshold *))
                    n_candidate_thresholds = get_candidate_thresholds(feature, X, y, leaf_samples,
                                                                      n_leaf_samples, &candidate_thresholds)

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

    cdef void get_leaf_samples(self,
                               Node*    node,
                               SIZE_t   remove_index,
                               SIZE_t** leaf_samples_ptr,
                               SIZE_t*  leaf_samples_count_ptr) nogil:
        """
        Recursively obtain the samples at the leaves and filter out deleted sample.
        """
        cdef bint   add_sample = True
        cdef SIZE_t i = 0
        cdef SIZE_t j = 0

        # leaf
        if node.is_leaf:

            # loop through all samples at this leaf
            for i in range(node.n_samples):

                # add sample to results if it has not been deleted
                if node.leaf_samples[i] != remove_index:
                    leaf_samples_ptr[0][leaf_samples_count_ptr[0]] = node.leaf_samples[i]
                    leaf_samples_count_ptr[0] += 1

        # decision node
        else:

            # traverse left
            if node.left:
                self.get_leaf_samples(node.left,
                                      remove_index,
                                      leaf_samples_ptr,
                                      leaf_samples_count_ptr)

            # traverse right
            if node.right:
                self.get_leaf_samples(node.right,
                                      remove_index,
                                      leaf_samples_ptr,
                                      leaf_samples_count_ptr)
