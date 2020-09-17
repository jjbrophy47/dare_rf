
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
    cdef int split_node(self, Node* node, int** X, int* y,
                        int* samples, int n_samples,
                        int topd, int min_support,
                        UINT32_t* random_state,
                        SplitRecord *split) nogil:
        """
        Splits the node by sampling from the valid feature distribution.
        Returns 0 for a successful split,
                1 to signal a leaf creation.
        """
        # parameters
        cdef bint use_gini = self.use_gini

        cdef double split_score = -1
        cdef int chosen_ndx = -1
        cdef double best_score = 1000000

        cdef int i
        cdef int j
        cdef int k

        cdef int result = 0

        if node.pos_count > 0 and node.pos_count < node.count:

            # exact, chooose best feaure
            if node.depth < topd or n_samples < min_support:
                best_score = 1000000
                chosen_ndx = -1

                for j in range(node.features_count):
                    split_score = compute_split_score(use_gini, node.count, node.left_counts[j],
                                                      node.right_counts[j], node.left_pos_counts[j],
                                                      node.right_pos_counts[j])

                    if split_score < best_score:
                        best_score = split_score
                        chosen_ndx = j

            # random, choose random feature
            else:
                chosen_ndx = int(rand_uniform(0, 1, random_state) / (1.0 / node.features_count))

            chosen_feature = node.features[chosen_ndx]

            # assign results from chosen feature
            split.left_indices = <int *>malloc(node.left_counts[chosen_ndx] * sizeof(int))
            split.right_indices = <int *>malloc(node.right_counts[chosen_ndx] * sizeof(int))
            j = 0
            k = 0
            for i in range(n_samples):
                if X[samples[i]][chosen_feature] == 1:
                    split.left_indices[j] = samples[i]
                    j += 1
                else:
                    split.right_indices[k] = samples[i]
                    k += 1
            split.left_count = j
            split.right_count = k
            split.feature = chosen_feature

            # add chosen feature to list of invalid nodes
            split.invalid_features_count = node.invalid_features_count + 1
            split.invalid_left_features = <int *>malloc(split.invalid_features_count * sizeof(int))
            split.invalid_right_features = <int *>malloc(split.invalid_features_count * sizeof(int))

            for i in range(node.invalid_features_count):
                split.invalid_left_features[i] = node.invalid_features[i]
                split.invalid_right_features[i] = node.invalid_features[i]

            split.invalid_left_features[split.invalid_features_count - 1] = chosen_feature
            split.invalid_right_features[split.invalid_features_count - 1] = chosen_feature

        # leaf
        else:
            result = 1

        return result


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int compute_splits(self, Node** node_ptr, int** X, int* y,
                            int* samples, int n_samples) nogil:
        """
        Update the metadata of this node.
        """
        cdef Node* node = node_ptr[0]

        cdef int count = n_samples
        cdef int pos_count = 0

        cdef int* left_counts = <int *>malloc(node.features_count * sizeof(int))
        cdef int* left_pos_counts = <int *>malloc(node.features_count * sizeof(int))
        cdef int* right_counts = <int *>malloc(node.features_count * sizeof(int))
        cdef int* right_pos_counts = <int *>malloc(node.features_count * sizeof(int))

        cdef int left_count
        cdef int left_pos_count

        cdef int i

        # count number of pos labels
        for i in range(n_samples):
            if y[samples[i]] == 1:
                pos_count += 1

        # compute statistics for each attribute
        for j in range(node.features_count):

            left_count = 0
            left_pos_count = 0

            for i in range(n_samples):
                if X[samples[i]][node.features[j]] == 1:
                    left_count += 1
                    left_pos_count += y[samples[i]]

            left_counts[j] = left_count
            left_pos_counts[j] = left_pos_count
            right_counts[j] = count - left_count
            right_pos_counts[j] = pos_count - left_pos_count

        node.count = count
        node.pos_count = pos_count
        node.left_counts = left_counts
        node.left_pos_counts = left_pos_counts
        node.right_counts = right_counts
        node.right_pos_counts = right_pos_counts

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void select_features(self, Node** node_ptr, int n_features, int n_max_features,
                              int* invalid_features, int n_invalid_features,
                              UINT32_t* random_state) nogil:
        """
        Select a random subset of features that are not alread used.
        """
        cdef Node* node = node_ptr[0]

        cdef int n_elem = n_max_features
        if n_features - n_invalid_features < n_max_features:
            n_elem = n_features - n_invalid_features

        cdef int ndx

        cdef int* new_arr = <int *>malloc(n_elem * sizeof(int))
        cdef int* generated_arr = <int *>malloc(n_elem * sizeof(int))

        cdef int i = 0
        cdef bint valid = True

        while i < n_elem:
            valid = True
            ndx = int(rand_uniform(0, 1, random_state) / (1.0 / n_features))

            # prevent repeating features
            for j in range(i):
                if ndx == generated_arr[j]:
                    valid = False
                    break

            # prevent already used features
            for j in range(n_invalid_features):
                if ndx == invalid_features[j]:
                    valid = False
                    break

            # add feature to feature pool
            if valid:
                new_arr[i] = ndx
                generated_arr[i] = ndx
                i += 1

        free(generated_arr)

        node.features = new_arr
        node.features_count = n_elem
        node.invalid_features = invalid_features
        node.invalid_features_count = n_invalid_features
