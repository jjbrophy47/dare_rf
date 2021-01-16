
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from ._tree cimport Feature
from ._tree cimport Threshold
from ._utils cimport compute_split_score
from ._utils cimport rand_uniform
from ._argsort cimport sort

# constants
cdef double UNDEF_LEAF_VAL = 0.5
cdef double FEATURE_THRESHOLD = 1e-7

cdef class _Splitter:
    """
    Splitter class.
    Finds the best splits on dense data, one split at a time.
    """

    def __cinit__(self,
                  SIZE_t  min_samples_leaf,
                  bint use_gini,
                  SIZE_t  k):
        """
        Parameters
        ----------
        min_samples_leaf : SIZE_t
            The minimal number of samples each leaf can have,
            where splits which would result in having less
            samples in a leaf are not considered.
        use_gini : bool
            If True, use the Gini index splitting criterion,
            otherwise entropy.
        k : SIZE_t
            Number of candidate thresholds to sample
            uniformly at random.
        """
        self.min_samples_leaf = min_samples_leaf
        self.use_gini = use_gini
        self.k = k

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void split_node(self,
                         Node**       node_ptr,
                         DTYPE_t**    X,
                         INT32_t*     y,
                         SIZE_t*      samples,
                         SIZE_t       n_samples,
                         SIZE_t       topd,
                         UINT32_t*    random_state,
                         SplitRecord* split) nogil:
        """
        Splits the node by sampling from the valid feature distribution.
        Returns 0 for a successful split,
                1 to signal a leaf creation.
        """
        cdef Node* node = node_ptr[0]

        # use Gini index if true, otherwise entropy
        cdef bint use_gini = self.use_gini

        # keep track of the best feature / threshold
        cdef DTYPE_t best_score = 1000000
        cdef DTYPE_t split_score = -1

        # object pointers
        cdef Feature*   feature = NULL
        cdef Threshold* threshold = NULL

        # save the best feature / threshold
        cdef INT32_t    chosen_feature_ndx = -1
        cdef INT32_t    chosen_threshold_ndx = -1
        cdef Feature*   chosen_feature = NULL
        cdef Threshold* chosen_threshold = NULL

        # iterators
        cdef SIZE_t i = 0
        cdef SIZE_t j = 0
        cdef SIZE_t k = 0

        # greedy node, chooose best feaure
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

                    # keep best score
                    if split_score < best_score:
                        best_score = split_score
                        chosen_feature = feature
                        chosen_threshold = threshold

        # random, choose random feature and threshold
        # TODO: is this random too good? Should we just be picking a random
        #       number inside the feature range?
        else:

            # select random feature
            chosen_feature_ndx = <INT32_t>(rand_uniform(0, 1, random_state) / (1.0 / node.n_features))
            chosen_feature = node.features[chosen_feature_ndx]

            # select random threshold from that feature
            chosen_threshold_ndx = <INT32_t>(rand_uniform(0, 1, random_state) / (1.0 / chosen_feature.n_thresholds))
            chosen_threshold = chosen_feature.thresholds[chosen_threshold_ndx]

        # split node samples based on the chosen feature / threshold
        split.left_samples = <SIZE_t *>malloc(chosen_threshold.n_left_samples * sizeof(SIZE_t))
        split.right_samples = <SIZE_t *>malloc(chosen_threshold.n_right_samples * sizeof(SIZE_t))
        j = 0
        k = 0
        for i in range(n_samples):
            if X[samples[i]][chosen_feature.index] <= chosen_threshold.value:
                split.left_samples[j] = samples[i]
                j += 1
            else:
                split.right_samples[k] = samples[i]
                k += 1
        split.n_left_samples = chosen_threshold.n_left_samples
        split.n_right_samples = chosen_threshold.n_right_samples

        # clear leaf node properties
        node.is_leaf = 0
        node.value = UNDEF_LEAF_VAL
        node.leaf_samples = NULL

        # set decision node properties
        node.chosen_feature = chosen_feature
        node.chosen_threshold = chosen_threshold


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef SIZE_t compute_metadata(self,
                                 Node**    node_ptr,
                                 DTYPE_t** X,
                                 INT32_t*  y,
                                 SIZE_t*   samples,
                                 SIZE_t    n_samples,
                                 UINT32_t* random_state) nogil:
        """
        For each feature:
          Sort the values,
          Identify ALL candidate thresholds.
            ->Reference: https://www.biostat.wisc.edu/~page/decision-trees.pdf
          Save metadata for each threshold.
        """
        cdef Node* node = node_ptr[0]

        # total number of positive samples
        cdef SIZE_t n_pos_samples = 0

        # keeps track of left and right branch info
        cdef SIZE_t count = 0
        cdef SIZE_t pos_count = 0
        cdef SIZE_t v_count = 0
        cdef SIZE_t v_pos_count = 0

        # keep track of the current feature set
        cdef DTYPE_t prev_val = -1
        cdef INT32_t prev_label = -1
        cdef DTYPE_t cur_val = -1
        cdef INT32_t cur_label = -1

        # helper arrays
        cdef DTYPE_t* values = <DTYPE_t *>malloc(n_samples * sizeof(DTYPE_t))
        cdef INT32_t* labels = <INT32_t *>malloc(n_samples * sizeof(INT32_t))
        cdef SIZE_t*  indices = <SIZE_t *>malloc(n_samples * sizeof(SIZE_t))

        # containers to hold feature value set information
        cdef DTYPE_t* threshold_values = <DTYPE_t *>malloc(n_samples * sizeof(DTYPE_t))
        cdef SIZE_t*  counts = <SIZE_t *>malloc(n_samples * sizeof(SIZE_t))
        cdef SIZE_t*  pos_counts = <SIZE_t *>malloc(n_samples * sizeof(SIZE_t))
        cdef SIZE_t*  v_counts = <SIZE_t *>malloc(n_samples * sizeof(SIZE_t))
        cdef SIZE_t*  v_pos_counts = <SIZE_t *>malloc(n_samples * sizeof(SIZE_t))

        # iterators
        cdef SIZE_t  i = 0
        cdef SIZE_t  k = 0
        cdef SIZE_t  j = 0
        cdef INT32_t ndx = 0

        # intermediate variables
        cdef DTYPE_t v1 = -1
        cdef DTYPE_t v2 = -1
        cdef DTYPE_t v1_label_ratio = -1
        cdef DTYPE_t v2_label_ratio = -1
        cdef bint    save_threshold = 0

        # threshold info to save
        cdef DTYPE_t value = -1
        cdef SIZE_t  v1_count = 0
        cdef SIZE_t  v1_pos_count = 0
        cdef SIZE_t  v2_count = 0
        cdef SIZE_t  v2_pos_count = 0
        cdef SIZE_t  left_count = 0
        cdef SIZE_t  left_pos_count = 0

        # counts
        cdef SIZE_t feature_value_count = 0
        cdef SIZE_t threshold_count = 0

        # container variables
        cdef SIZE_t      feature_index = 0
        cdef Threshold*  threshold = NULL
        cdef Threshold** thresholds = NULL
        cdef SIZE_t      n_thresholds = 0
        cdef SIZE_t      k_samples = self.k

        # variables for sampling
        cdef SIZE_t*     sampled_indices = NULL
        cdef Threshold** sampled_thresholds = NULL
        cdef Feature*    feature = NULL
        cdef bint        valid = True

        # return variable
        cdef SIZE_t n_usable_thresholds = 0

        # count number of pos labels
        for i in range(n_samples):
            labels[i] = y[samples[i]]
            if y[samples[i]] == 1:
                n_pos_samples += 1

        # printf('labels[20308]: %d\n', labels[20308])

        # printf('n_samples: %ld\n', n_samples)

        # printf('n_features: %ld\n', node.n_features)

        # compute statistics for each attribute
        for j in range(node.n_features):

            # PART 1: compute unique feature value set information

            # access feature object
            feature = node.features[j]

            # printf('\n[CM - PART 1] feature %ld\n', feature.index)

            # copy values and indices for this feature
            for i in range(n_samples):
                values[i] = X[samples[i]][feature.index]
                indices[i] = i

            # sort feature values
            # printf('sorting\n')
            sort(values, indices, n_samples)
            # printf('sorted\n')

            # for i in range(n_samples):
            #     printf('values[%ld]: %.2f, indices[%ld]: %ld, labels[%ld]: %d\n',
            #            i, values[i], i, indices[i], i, labels[indices[i]])

            # check to see if this feature is constant
            # printf('vlast: %.2f, vfirst: %.2f\n', values[n_samples - 1], values[0])
            if values[n_samples - 1] <= values[0] + FEATURE_THRESHOLD:
                feature.thresholds = NULL
                feature.n_thresholds = 0
                continue

            # initialize starting values
            count = 1
            pos_count = labels[indices[0]]
            v_count = 1
            v_pos_count = labels[indices[0]]
            feature_value_count = 0
            prev_val = values[0]
            prev_label = labels[indices[0]]

            # printf('prev_label: %d\n', prev_label)

            # result container
            thresholds = <Threshold **>malloc(n_samples * sizeof(Threshold *))

            # loop through sorted feature values
            for i in range(1, n_samples):
                # printf('i: %ld\n', i)
                # printf('  indices[%ld]: %ld\n', i, indices[i])
                # printf('    labels[%ld]: %d\n', indices[i], labels[indices[i]])
                cur_val = values[i]
                cur_label = labels[indices[i]]

                # next feature value
                if cur_val > prev_val + FEATURE_THRESHOLD:

                    # save previous feature counts
                    threshold_values[feature_value_count] = prev_val
                    counts[feature_value_count] = count
                    pos_counts[feature_value_count] = pos_count
                    v_counts[feature_value_count] = v_count
                    v_pos_counts[feature_value_count] = v_pos_count
                    feature_value_count += 1

                    # reset counts for this new feature
                    v_count = 1
                    v_pos_count = cur_label

                # same feature value
                else:

                    # increment counts for this feature
                    v_count += 1
                    v_pos_count += cur_label

                # increment left branch counts
                count += 1
                pos_count += cur_label

                # move pointers to the next feature
                prev_val = cur_val
                prev_label = cur_label

            # handle last feature value
            if v_count > 0:

                # save previous feature counts
                threshold_values[feature_value_count] = prev_val
                counts[feature_value_count] = count
                pos_counts[feature_value_count] = pos_count
                v_counts[feature_value_count] = v_count
                v_pos_counts[feature_value_count] = v_pos_count
                feature_value_count += 1

            # printf('[CM - PART 1] no. feature value sets: %ld\n', feature_value_count)

            # PART 2: evaluate each pair of feature sets to get candidate thresholds
            threshold_count = 0
            for k in range(1, feature_value_count):

                # extract both of the feature set counts
                v1 = threshold_values[k-1]
                v2 = threshold_values[k]
                v1_count = v_counts[k-1]
                v2_count = v_counts[k]
                v1_pos_count = v_pos_counts[k-1]
                v2_pos_count = v_pos_counts[k]
                left_count = counts[k-1]
                left_pos_count = pos_counts[k-1]

                # compute label ratios of the two groups
                v1_label_ratio = v1_pos_count / (1.0 * v1_count)
                v2_label_ratio = v2_pos_count / (1.0 * v2_count)

                save_threshold = ((v1_label_ratio != v2_label_ratio) or
                                  (v1_label_ratio > 0.0 and v1_label_ratio < 1.0) or
                                  (v2_label_ratio > 0.0 and v2_label_ratio < 1.0))

                # save threshold
                if save_threshold:

                    # create threshold
                    threshold = <Threshold *>malloc(sizeof(Threshold))
                    threshold.value = (v1 + v2) / 2.0
                    threshold.n_v1_samples = v1_count
                    threshold.n_v1_pos_samples = v1_pos_count
                    threshold.n_v2_samples = v2_count
                    threshold.n_v2_pos_samples = v2_pos_count
                    threshold.n_left_samples = left_count
                    threshold.n_left_pos_samples = left_pos_count
                    threshold.n_right_samples = n_samples - left_count
                    threshold.n_right_pos_samples = n_pos_samples - left_pos_count

                    # save threshold to this feature
                    thresholds[threshold_count] = threshold
                    threshold_count += 1
                    n_usable_thresholds += 1

                    # printf('[CM - PART 2] candidate threshold.value: %.2f\n', threshold.value)
            # printf('[CM - PART 2] no. candidate thresholds: %ld\n', threshold_count)

            # PART 3: sample k candidate thresholds uniformly at random
            if threshold_count < k_samples:
                n_thresholds = threshold_count
            else:
                n_thresholds = k_samples

            # create new (smaller) thresholds array
            sampled_thresholds = <Threshold **>malloc(n_thresholds * sizeof(Threshold *))
            sampled_indices = <SIZE_t *>malloc(n_thresholds * sizeof(SIZE_t))

            # sample threshold indices uniformly at random
            i = 0
            while i < n_thresholds:
                valid = True

                # sample feature index
                ndx = <INT32_t>(rand_uniform(0, 1, random_state) / (1.0 / threshold_count))

                # invalid: already sampled
                for k in range(i):
                    if ndx == sampled_indices[k]:
                        valid = False
                        break

                # valid: add threshold to sampled list of candidate thresholds
                if valid:

                    # copy threshold
                    threshold = thresholds[ndx]
                    t2 = <Threshold *>malloc(sizeof(Threshold))
                    t2.value = threshold.value
                    t2.n_v1_samples = threshold.n_v1_samples
                    t2.n_v1_pos_samples = threshold.n_v1_pos_samples
                    t2.n_v2_samples = threshold.n_v2_samples
                    t2.n_v2_pos_samples = threshold.n_v2_pos_samples
                    t2.n_left_samples = threshold.n_left_samples
                    t2.n_left_pos_samples = threshold.n_left_pos_samples
                    t2.n_right_samples = threshold.n_right_samples
                    t2.n_right_pos_samples = threshold.n_right_pos_samples

                    # add copied threshold to pool
                    sampled_thresholds[i] = t2
                    sampled_indices[i] = ndx
                    i += 1

                    # printf('[CM - PART 3] sampled threshold.value: %.2f\n', t2.value)
            # printf('[CM - PART 3] no. sampled thresholds: %ld\n', i)

            # set thresholds property on this feature
            feature.thresholds = sampled_thresholds
            feature.n_thresholds = n_thresholds

            # free thresholds array and its contents
            for i in range(threshold_count):
                free(thresholds[i])
            free(thresholds)
            free(sampled_indices)

        # clean up
        free(values)
        free(labels)
        free(indices)

        free(threshold_values)
        free(counts)
        free(pos_counts)
        free(v_counts)
        free(v_pos_counts)

        return n_usable_thresholds

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void select_features(self,
                              Node**    node_ptr,
                              SIZE_t    n_total_features,
                              SIZE_t    n_max_features,
                              UINT32_t* random_state) nogil:
        """
        Select a random subset of features that are not alread used.
        """
        cdef Node* node = node_ptr[0]

        # get number of features to sample
        cdef SIZE_t n_elem = n_max_features
        if n_total_features < n_max_features:
            n_elem = n_total_features

        cdef INT32_t ndx = 0

        cdef Feature*  feature = NULL
        cdef Feature** features = <Feature **>malloc(n_elem * sizeof(Feature*))
        cdef SIZE_t*   sampled_indices = <SIZE_t *>malloc(n_elem * sizeof(SIZE_t))

        cdef SIZE_t i = 0
        cdef bint valid = True

        # sample feature indices uniformly at random
        while i < n_elem:
            valid = True

            # sample feature index
            ndx = <INT32_t>(rand_uniform(0, 1, random_state) / (1.0 / n_total_features))

            # invalid: already sampled
            for j in range(i):
                if ndx == sampled_indices[j]:
                    valid = False
                    break

            # valid: create feature and add it to the pool
            if valid:
                feature = <Feature *>malloc(sizeof(Feature))
                feature.index = ndx

                features[i] = feature
                sampled_indices[i] = ndx
                i += 1

        free(sampled_indices)

        node.features = features
        node.n_features = n_elem
