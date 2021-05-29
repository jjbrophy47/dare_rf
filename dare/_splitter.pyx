# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
"""
Splits the data into separate partitions.
"""
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport compute_split_score
from ._utils cimport rand_uniform
from ._utils cimport rand_int
from ._utils cimport create_intlist
from ._utils cimport create_feature
from ._utils cimport create_threshold
from ._utils cimport free_intlist
from ._utils cimport copy_intlist
from ._utils cimport copy_feature
from ._utils cimport copy_threshold
from libc.math cimport fabs
from ._argsort cimport sort

# constants
cdef DTYPE_t FEATURE_THRESHOLD = 0.0000001

cdef class _Splitter:
    """
    Splitter class.
    Finds the best splits on dense data, one split at a time.
    """

    def __cinit__(self,
                  _Config config):
        """
        Consructor.
        """
        self.config = config

    cdef SIZE_t select_threshold(self,
                                 Node*        node,
                                 DTYPE_t**    X,
                                 INT32_t*     y,
                                 IntList*     samples,
                                 SIZE_t       n_total_features) nogil:
        """
        Select a threshold and save metadata to make future updates efficient.
        """
        cdef SIZE_t n_usable_thresholds = 0

        # greedy node
        if node.depth >= self.config.topd:
            n_usable_thresholds = select_greedy_threshold(node, X, y, samples, n_total_features, self.config)

        # random node
        else:
            n_usable_thresholds = select_random_threshold(node, X, samples, n_total_features, self.config)

        return n_usable_thresholds


cdef SIZE_t select_greedy_threshold(Node*     node,
                                    DTYPE_t** X,
                                    INT32_t*  y,
                                    IntList*  samples,
                                    SIZE_t    n_total_features,
                                    _Config   config) nogil:
    """
    Select threshold for a greedy node, and save the metadata.

    For each feature:
      -Sort the values.
      -Identify ALL candidate thresholds:
        ->Reference: https://www.biostat.wisc.edu/~page/decision-trees.pdf
      -Save metadata for each threshold.
    """

    # configuration
    cdef SIZE_t    topd = config.topd
    cdef SIZE_t    k_samples = config.k
    cdef SIZE_t    max_features = config.max_features
    cdef SIZE_t    min_samples_leaf = config.min_samples_leaf
    cdef bint      use_gini = config.use_gini
    cdef UINT32_t* random_state = &config.rand_r_state

    # iterators
    cdef SIZE_t  i = 0
    cdef SIZE_t  j = 0
    cdef SIZE_t  k = 0

    # samplers
    cdef INT32_t feature_index = 0
    cdef INT32_t ndx = 0

    # get number of features to sample
    cdef Feature** features = NULL
    cdef SIZE_t    n_features = 0

    # container variables
    cdef Feature*    feature = NULL
    cdef Threshold*  threshold = NULL
    cdef Threshold** candidate_thresholds = NULL
    cdef SIZE_t      n_candidate_thresholds = 0
    cdef SIZE_t      n_candidate_thresholds_to_sample = 0

    # helper arrays
    cdef DTYPE_t* values = <DTYPE_t *>malloc(samples.n * sizeof(DTYPE_t))
    cdef INT32_t* labels = <INT32_t *>malloc(samples.n * sizeof(INT32_t))
    cdef SIZE_t*  indices = <SIZE_t *>malloc(samples.n * sizeof(SIZE_t))
    cdef SIZE_t   n_pos_samples = 0

    # variables for tracking
    cdef IntList*    constant_features = NULL
    cdef IntList*    sampled_features = NULL
    cdef IntList*    sampled_indices = NULL
    cdef Threshold** sampled_thresholds = NULL
    cdef bint        valid = True

    # keep track of the best feature / threshold
    cdef DTYPE_t    best_score = 1000000
    cdef DTYPE_t    split_score = -1
    cdef Feature*   chosen_feature = NULL
    cdef Threshold* chosen_threshold = NULL

    # return variable
    cdef SIZE_t n_usable_thresholds = 0

    # compute no. pos. samples
    for i in range(samples.n):
        labels[i] = y[samples.arr[i]]
        n_pos_samples += y[samples.arr[i]]

    # allocate memory for a features array
    if n_total_features < max_features:
        max_features = n_total_features
    features = <Feature **>malloc(max_features * sizeof(Feature *))

    # variables to keep track of constant and sampled features
    constant_features = copy_intlist(node.constant_features, n_total_features)
    sampled_features = create_intlist(n_total_features, 0)

    # sample features until `max_features` is reached or there are no features left
    while n_features < max_features and (sampled_features.n + constant_features.n) < n_total_features:

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

        # printf('[S - SGT] feature_index: %d\n', feature_index)

        # copy values and labels into new arrays, and count no. pos. labels
        for i in range(samples.n):
            values[i] = X[samples.arr[i]][feature_index]
            indices[i] = i

        # sort feature values, and their corresponding indices
        sort(values, indices, samples.n)

        # constant feature
        if values[samples.n - 1] <= values[0] + FEATURE_THRESHOLD:
            constant_features.arr[constant_features.n] = feature_index
            constant_features.n += 1
            continue

        # add feature index to list of sampled features
        else:
            sampled_features.arr[sampled_features.n] = feature_index
            sampled_features.n += 1

        # get candidate thresholds for this feature
        candidate_thresholds = <Threshold **>malloc(samples.n * sizeof(Threshold *))
        n_candidate_thresholds = get_candidate_thresholds(values, labels, indices, samples.n,
                                                          n_pos_samples, min_samples_leaf,
                                                          &candidate_thresholds)

        # no valid thresholds, candidate thresholds is freed in the method
        if n_candidate_thresholds == 0:
            continue

        # increment total no. of valid thresholds
        n_usable_thresholds += n_candidate_thresholds

        # create feature object
        feature = <Feature *>malloc(sizeof(Feature))
        feature.index = feature_index

        # printf('[S - SGT] feature_index: %d\n', feature_index)

        # sample k candidate thresholds uniformly at random
        if n_candidate_thresholds < k_samples:
            n_candidate_thresholds_to_sample = n_candidate_thresholds
        else:
            n_candidate_thresholds_to_sample = k_samples

        # printf('[S - SGT] n_candidate_thresholds_to_sample: %ld\n', n_candidate_thresholds_to_sample)

        # create new (smaller) thresholds array
        final_thresholds = <Threshold **>malloc(n_candidate_thresholds_to_sample * sizeof(Threshold *))
        sampled_indices = create_intlist(n_candidate_thresholds_to_sample, 0)

        # sample threshold indices uniformly at random
        while sampled_indices.n < n_candidate_thresholds_to_sample:
            valid = True

            # sample a threshold index
            ndx = rand_int(0, n_candidate_thresholds, random_state)

            # invalid: already sampled
            for i in range(sampled_indices.n):
                if ndx == sampled_indices.arr[i]:
                    valid = False
                    break

            # valid threshold
            if valid:

                # add copied threshold to thresholds array
                threshold = copy_threshold(candidate_thresholds[ndx])
                final_thresholds[sampled_indices.n] = threshold
                sampled_indices.arr[sampled_indices.n] = ndx
                sampled_indices.n += 1

                # compute split score
                split_score = compute_split_score(use_gini,
                                                  node.n_samples,
                                                  threshold.n_left_samples,
                                                  threshold.n_right_samples,
                                                  threshold.n_left_pos_samples,
                                                  threshold.n_right_pos_samples)

                # save if its the best score
                if split_score < best_score:
                    best_score = split_score
                    chosen_feature = feature
                    chosen_threshold = threshold
                    # printf('[S - SGT] score: %.5f\n', best_score)
                    # printf('[S - SGT] chosen_feature.index: %ld, chosen_threshold.value: %.5f\n',
                    #        chosen_feature.index, chosen_threshold.value)
                    # printf('[S - SGT] threshold.n_left_samples: %ld, threshold.n_right_samples: %ld\n',
                    #        threshold.n_left_samples, threshold.n_right_samples)

        # printf('[S - SGT] n_candidate_thresholds_to_sample: %ld\n', n_candidate_thresholds_to_sample)

        # set threshold properties for this feature
        feature.thresholds = final_thresholds
        feature.n_thresholds = n_candidate_thresholds_to_sample

        # add feature to features array
        features[n_features] = feature
        n_features += 1

        # free candidate thresholds array
        for i in range(n_candidate_thresholds):
            free(candidate_thresholds[i])
        free(candidate_thresholds)

        # free sampled indices array
        free_intlist(sampled_indices)

    # free previous constant features array
    free_intlist(node.constant_features)

    # set node properties
    if n_usable_thresholds > 0:
        node.features = <Feature **>realloc(features, n_features * sizeof(Feature *))
        node.n_features = n_features
        node.constant_features = copy_intlist(constant_features, constant_features.n)
        node.chosen_feature = copy_feature(chosen_feature)
        node.chosen_threshold = copy_threshold(chosen_threshold)

    # free features array
    else:
        node.constant_features = copy_intlist(constant_features, constant_features.n)
        free(features)

    # printf('[R - SNF] node.n_features: %ld\n', node.n_features)

    # free constant features
    free_intlist(constant_features)

    # clean up
    free(values)
    free(labels)
    free(indices)
    free_intlist(sampled_features)

    return n_usable_thresholds


cdef SIZE_t select_random_threshold(Node*     node,
                                    DTYPE_t** X,
                                    IntList*  samples,
                                    SIZE_t    n_total_features,
                                    _Config   config) nogil:
    """
    "Extremely Randomized Node" (k=1).
    Reference: https://link.springer.com/article/10.1007/s10994-006-6226-1

    -Select a feature at random.
    -Compute min. and max. values.
    -Select a threshold between [min, max], uniformly at random.
    -Repeat if feature is constant or partition does not meet min. no. leaf samples.
    """

    # configuration
    cdef SIZE_t    min_samples_leaf = config.min_samples_leaf
    cdef UINT32_t* random_state = &config.rand_r_state

    # keep track of the current feature value
    cdef DTYPE_t cur_val = 0
    cdef DTYPE_t min_val = 0
    cdef DTYPE_t max_val = 0

    # arrays to keep track of invalid features
    cdef IntList* constant_features = copy_intlist(node.constant_features, n_total_features)
    cdef IntList* sampled_features = create_intlist(n_total_features, 0)

    # feature / threshold information
    cdef Feature*   feature = NULL
    cdef Threshold* threshold = NULL
    cdef DTYPE_t    threshold_value = 0
    cdef SIZE_t     n_left_samples = 0
    cdef SIZE_t     n_right_samples = 0

    # counters
    cdef SIZE_t i = 0

    # return variable
    cdef SIZE_t n_usable_thresholds = 0

    # sample features until a valid one is found or there are no more features
    while sampled_features.n + constant_features.n < n_total_features:

        # sample feature index
        feature_index = rand_int(0, n_total_features, random_state)

        # check if feature has already been sampled
        valid = True
        for i in range(sampled_features.n):
            if sampled_features.arr[i] == feature_index:
                valid = False
                break
        if not valid:
            continue

        # check if feature is a known constant
        valid = True
        for i in range(constant_features.n):
            if constant_features.arr[i] == feature_index:
                valid = False
                break
        if not valid:
            continue

        # find min. max. values, and their counts
        min_val = X[samples.arr[0]][feature_index]
        max_val = min_val

        for i in range(samples.n):
            cur_val = X[samples.arr[i]][feature_index]

            if cur_val < min_val:
                min_val = cur_val

            elif cur_val > max_val:
                max_val = cur_val

        # constant feature
        if max_val <= min_val + FEATURE_THRESHOLD:
            constant_features.arr[constant_features.n] = feature_index
            constant_features.n += 1

        # non-constant feature
        else:

            # add feature to list of sampled features
            sampled_features.arr[sampled_features.n] = feature_index
            sampled_features.n += 1

            # keep randomly sampling until a valid threshold is found
            threshold_value = <DTYPE_t>rand_uniform(min_val, max_val, random_state)
            while threshold_value >= max_val or threshold_value < min_val:
                threshold_value = <DTYPE_t>rand_uniform(min_val, max_val, random_state)

            # make sure the min. no. samples is met for both branches
            n_left_samples = 0
            n_right_samples = 0

            for i in range(samples.n):

                # increment sample count for left or right branch
                if X[samples.arr[i]][feature_index] <= threshold_value:
                    n_left_samples += 1

                else:
                    n_right_samples += 1

            # free previous constant thresholds array
            free_intlist(node.constant_features)

            # save node properties
            # printf('saving node\n')
            node.chosen_feature = create_feature(feature_index)
            node.chosen_threshold = create_threshold(threshold_value, n_left_samples, n_right_samples)
            node.constant_features = copy_intlist(constant_features, constant_features.n)

            # free constant features
            free_intlist(constant_features)

            # increment no. usable thresholds
            n_usable_thresholds += 1

            # exit while loop
            break

    # clean up
    free_intlist(sampled_features)

    return n_usable_thresholds


cdef SIZE_t get_candidate_thresholds(DTYPE_t*     values,
                                     INT32_t*     labels,
                                     SIZE_t*      indices,
                                     SIZE_t       n_samples,
                                     SIZE_t       n_pos_samples,
                                     SIZE_t       min_samples_leaf,
                                     Threshold*** thresholds_ptr) nogil:
    """
    For this feature:

      -Sort the values.
      -Identify ALL candidate thresholds.
        ->Reference: https://www.biostat.wisc.edu/~page/decision-trees.pdf
      -Save metadata for each threshold.

    NOTE: values and indices were sorted together, labels was not!
    """

    # keeps track of left and right branch info
    cdef SIZE_t count = 1
    cdef SIZE_t pos_count = labels[indices[0]]
    cdef SIZE_t v_count = 1
    cdef SIZE_t v_pos_count = labels[indices[0]]

    # keep track of the current feature set
    cdef DTYPE_t prev_val = values[0]
    cdef DTYPE_t cur_val = 0
    cdef INT32_t cur_label = 0

    # containers to hold feature value set information
    cdef DTYPE_t* threshold_values = <DTYPE_t *>malloc(n_samples * sizeof(DTYPE_t))
    cdef SIZE_t*  counts = <SIZE_t *>malloc(n_samples * sizeof(SIZE_t))
    cdef SIZE_t*  pos_counts = <SIZE_t *>malloc(n_samples * sizeof(SIZE_t))
    cdef SIZE_t*  v_counts = <SIZE_t *>malloc(n_samples * sizeof(SIZE_t))
    cdef SIZE_t*  v_pos_counts = <SIZE_t *>malloc(n_samples * sizeof(SIZE_t))

    # iterators
    cdef SIZE_t  i = 0
    cdef SIZE_t  k = 0

    # intermediate variables
    cdef DTYPE_t v1_label_ratio = 0
    cdef DTYPE_t v2_label_ratio = 0
    cdef bint    save_threshold = False

    # threshold info to save
    cdef DTYPE_t value = 0
    cdef DTYPE_t v1 = 0
    cdef DTYPE_t v2 = 0
    cdef SIZE_t  n_v1_samples = 0
    cdef SIZE_t  n_v1_pos_samples = 0
    cdef SIZE_t  n_v2_samples = 0
    cdef SIZE_t  n_v2_pos_samples = 0
    cdef SIZE_t  n_left_samples = 0
    cdef SIZE_t  n_left_pos_samples = 0
    cdef SIZE_t  n_right_samples = 0
    cdef SIZE_t  n_right_pos_samples = 0

    # counts
    cdef SIZE_t feature_value_count = 0
    cdef SIZE_t thresholds_count = 0

    # object pointers
    cdef Threshold*  threshold = NULL
    cdef Threshold** thresholds = thresholds_ptr[0]

    # save statistics about each adjacent feature value
    for i in range(1, n_samples):
        cur_val = values[i]
        cur_label = labels[indices[i]]

        # same feature, increment counts
        # if fabs(cur_val - prev_val) <= FEATURE_THRESHOLD:
        if cur_val <= prev_val + FEATURE_THRESHOLD:
            # printf('[S - GCT] %.32f, %.32f\n', cur_val, prev_val)
            # printf('[S - GCT] %.32f <= %.32f\n', fabs(cur_val - prev_val), FEATURE_THRESHOLD)
            v_count += 1
            v_pos_count += cur_label

        # next feature:
        else:

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

        # increment left branch counts
        count += 1
        pos_count += cur_label

        # move pointers to the next feature
        prev_val = cur_val

    # handle last feature value
    if v_count > 0:

        # save previous feature counts
        threshold_values[feature_value_count] = prev_val
        counts[feature_value_count] = count
        pos_counts[feature_value_count] = pos_count
        v_counts[feature_value_count] = v_count
        v_pos_counts[feature_value_count] = v_pos_count
        feature_value_count += 1

    # evaluate adjacent pairs of feature sets to get candidate thresholds
    for k in range(1, feature_value_count):

        # extract both of the feature set counts
        v1 = threshold_values[k-1]
        v2 = threshold_values[k]
        n_v1_samples = v_counts[k-1]
        n_v2_samples = v_counts[k]
        n_v1_pos_samples = v_pos_counts[k-1]
        n_v2_pos_samples = v_pos_counts[k]
        n_left_samples = counts[k-1]
        n_left_pos_samples = pos_counts[k-1]
        n_right_samples = n_samples - n_left_samples
        n_right_pos_samples = n_pos_samples - n_left_pos_samples

        # if n_left_samples < min_samples_leaf or n_right_samples < min_samples_leaf:
        #     printf('[S - GCT] NO LEFT OR RIGHT SAMPLES\n')

        # not enough samples in each branch
        if n_left_samples < min_samples_leaf or n_right_samples < min_samples_leaf:
            continue

        # compute label ratios of the two groups
        v1_label_ratio = n_v1_pos_samples / (1.0 * n_v1_samples)
        v2_label_ratio = n_v2_pos_samples / (1.0 * n_v2_samples)

        # valid threshold
        if ((v1_label_ratio != v2_label_ratio) or
            (v1_label_ratio > 0.0 and v2_label_ratio < 1.0)):

            # create threshold
            threshold = <Threshold *>malloc(sizeof(Threshold))
            threshold.v1 = v1
            threshold.v2 = v2
            threshold.value = v1
            threshold.n_v1_samples = n_v1_samples
            threshold.n_v1_pos_samples = n_v1_pos_samples
            threshold.n_v2_samples = n_v2_samples
            threshold.n_v2_pos_samples = n_v2_pos_samples
            threshold.n_left_samples = n_left_samples
            threshold.n_left_pos_samples = n_left_pos_samples
            threshold.n_right_samples = n_right_samples
            threshold.n_right_pos_samples = n_right_pos_samples

            # printf('[S - GCT] v1: %.5f, v2: %.5f, v1 + v2: %.5f, threshold.value: %.5f\n',
            #        v1, v2, v2 + v1, threshold.value)

            # save threshold to thresholds array
            thresholds[thresholds_count] = threshold
            thresholds_count += 1

    # if no viable thresholds, free thresholds array container
    if thresholds_count == 0:
        free(thresholds)
        thresholds_ptr[0] = NULL

    # clean up
    free(threshold_values)
    free(counts)
    free(pos_counts)
    free(v_counts)
    free(v_pos_counts)

    return thresholds_count
