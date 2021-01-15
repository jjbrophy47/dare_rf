
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
from ._argsort cimport argsort

# constants
cdef double UNDEF_LEAF_VAL = 0.5

cdef class _Splitter:
    """
    Splitter class.
    Finds the best splits on dense data, one split at a time.
    """

    def __cinit__(self, int min_samples_leaf, bint use_gini):
        """
        Parameters
        ----------
        min_samples_leaf : int
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not considered.
        use_gini : bool
            If True, use the Gini index splitting criterion; otherwise
            use entropy.
        """
        self.min_samples_leaf = min_samples_leaf
        self.use_gini = use_gini

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int split_node(self,
                        Node**       node_ptr,
                        double**     X,
                        int*         y,
                        int*         samples,
                        int          n_samples,
                        int          topd,
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
        cdef double best_score = 1000000
        cdef double split_score = -1

        # object pointers
        cdef Feature*   feature = NULL
        cdef Threshold* threshold = NULL

        # save the best feature / threshold
        cdef int        chosen_feature_ndx = -1
        cdef int        chosen_threshold_ndx = -1
        cdef Feature*   chosen_feature = NULL
        cdef Threshold* chosen_threshold = NULL

        # iterators
        cdef int i = 0
        cdef int j = 0
        cdef int k = 0

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
            chosen_feature_ndx = int(rand_uniform(0, 1, random_state) / (1.0 / node.n_features))
            chosen_feature = node.features[chosen_feature_ndx]

            # select random threshold from that feature
            chosen_threshold_ndx = int(rand_uniform(0, 1, random_state) / (1.0 / chosen_feature.n_thresholds))
            chosen_threshold = chosen_feature.thresholds[chosen_threshold_ndx]

        # split node samples based on the chosen feature / threshold
        split.left_samples = <int *>malloc(chosen_threshold.n_left_samples * sizeof(int))
        split.right_samples = <int *>malloc(chosen_threshold.n_right_samples * sizeof(int))
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
    cdef int compute_metadata(self,
                              Node**   node_ptr,
                              double** X,
                              int*     y,
                              int*     samples,
                              int      n_samples) nogil:
        """
        For each feature:
          Sort the values,
          Identify ALL candidate thresholds.
            ->Reference: https://www.biostat.wisc.edu/~page/decision-trees.pdf
          Save metadata for each threshold.
        """
        cdef Node* node = node_ptr[0]

        # total number of positive samples
        cdef int n_pos_samples = 0

        # keeps track of left and right branch info
        cdef int count = -1
        cdef int pos_count = -1
        cdef int v_count = -1
        cdef int v_pos_count = -1

        # keep track of the current feature set
        cdef double prev_val = -1
        cdef int    prev_label = -1
        cdef double cur_val = -1
        cdef int    cur_label = -1

        # helper arrays
        cdef double* values = <double *>malloc(n_samples * sizeof(double))
        cdef int*    labels = <int *>malloc(n_samples * sizeof(int))
        cdef int*    indices = <int *>malloc(n_samples * sizeof(int))

        # containers to hold feature value set information
        cdef double* threshold_values = <double *>malloc(n_samples * sizeof(double))
        cdef int*    counts = <int *>malloc(n_samples * sizeof(int))
        cdef int*    pos_counts = <int *>malloc(n_samples * sizeof(int))
        cdef int*    v_counts = <int *>malloc(n_samples * sizeof(int))
        cdef int*    v_pos_counts = <int *>malloc(n_samples * sizeof(int))

        # iterators
        cdef int i = 0
        cdef int k = 0
        cdef int j = 0
        cdef int threshold_count = 0
        cdef int ndx = 0

        # intermediate variables
        cdef double v1 = -1
        cdef double v2 = -1
        cdef double v1_label_ratio = -1
        cdef double v2_label_ratio = -1
        cdef bint save_threshold = 0

        # threshold info to save
        cdef double value = -1
        cdef int v1_count = -1
        cdef int v1_pos_count = -1
        cdef int v2_count = -1
        cdef int v2_pos_count = -1
        cdef int left_count = -1
        cdef int left_pos_count = -1

        # pointer variables
        cdef Feature*   feature = NULL
        cdef Threshold* threshold = NULL

        # count number of pos labels
        for i in range(n_samples):
            labels[i] = y[samples[i]]
            if y[samples[i]] == 1:
                n_pos_samples += 1

        # compute statistics for each attribute
        for j in range(node.n_features):

            # access feature object
            feature = node.features[j]
            feature.thresholds = <Threshold **>malloc(n_samples * sizeof(Threshold*))
            feature.n_thresholds = 0

            # copy values for this feature
            for i in range(n_samples):
                values[i] = X[samples[i]][feature.index]

            # sort feature values
            argsort(values, indices, n_samples)

            # initialize starting values
            count = 1
            pos_count = labels[indices[0]]
            v_count = 1
            v_pos_count = labels[indices[0]]
            threshold_count = 0
            prev_val = values[indices[0]]
            prev_label = labels[indices[0]]

            # loop through sorted feature values
            for i in range(1, n_samples):
                ndx = indices[i]
                cur_val = values[ndx]
                cur_label = labels[ndx]

                # next feature value
                if cur_val > prev_val + 1e-7:

                    # save previous feature counts
                    threshold_values[threshold_count] = prev_val
                    counts[threshold_count] = count
                    pos_counts[threshold_count] = pos_count
                    v_counts[threshold_count] = v_count
                    v_pos_counts[threshold_count] = v_pos_count
                    threshold_count += 1

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
                threshold_values[threshold_count] = prev_val
                counts[threshold_count] = count
                pos_counts[threshold_count] = pos_count
                v_counts[threshold_count] = v_count
                v_pos_counts[threshold_count] = v_pos_count
                threshold_count += 1

            # printf('[CM] no. feature value sets: %d\n', threshold_count)

            # evaluate each pair of feature sets
            for k in range(1, threshold_count):

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
                    feature.thresholds[feature.n_thresholds] = threshold
                    feature.n_thresholds += 1

                    # printf('[CM] feature: %d, threshold.value: %.2f\n', feature.index, threshold.value)

            # adjust thresholds to appropriate length
            feature.thresholds = <Threshold **>realloc(feature.thresholds,
                                                       feature.n_thresholds * sizeof(Threshold *))

        # clean up
        free(values)
        free(labels)
        free(indices)

        free(threshold_values)
        free(counts)
        free(pos_counts)
        free(v_counts)
        free(v_pos_counts)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void select_features(self,
                              Node** node_ptr,
                              int n_total_features,
                              int n_max_features,
                              UINT32_t* random_state) nogil:
        """
        Select a random subset of features that are not alread used.
        """
        cdef Node* node = node_ptr[0]

        # get number of features to sample
        cdef int n_elem = n_max_features
        if n_total_features < n_max_features:
            n_elem = n_total_features

        cdef int ndx = 0

        cdef Feature*  feature = NULL
        cdef Feature** features = <Feature **>malloc(n_elem * sizeof(Feature*))
        cdef int*      sample_indices = <int *>malloc(n_elem * sizeof(int))

        cdef int i = 0
        cdef bint valid = True

        # sample feature indices uniformly at random
        while i < n_elem:
            valid = True

            # sample feature index
            ndx = int(rand_uniform(0, 1, random_state) / (1.0 / n_total_features))

            # invalid: already sampled
            for j in range(i):
                if ndx == sample_indices[j]:
                    valid = False
                    break

            # valid: create feature and add it to the pool
            if valid:
                feature = <Feature *>malloc(sizeof(Feature))
                feature.index = ndx

                features[i] = feature
                sample_indices[i] = ndx
                i += 1

        free(sample_indices)

        node.features = features
        node.n_features = n_elem
