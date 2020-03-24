
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport get_random
from ._utils cimport compute_gini
from ._utils cimport generate_distribution
from ._utils cimport sample_distribution

cdef class _Splitter:
    """
    Splitter class.
    Finds the best splits on dense data, one split at a time.
    """

    def __cinit__(self, int min_samples_leaf, double lmbda):
        """
        Parameters
        ----------
        min_samples_leaf : int
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.
        lmbda : double
            Noise control when generating distribution; higher values mean a
            more deterministic algorithm.
        """
        self.min_samples_leaf = min_samples_leaf
        self.lmbda = lmbda

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int split_node(self, Node* node, int** X, int* y,
                        int* samples, int n_samples, double parent_p,
                        SplitRecord *split) nogil:
        """
        Splits the node by sampling from the valid feature distribution.
        Returns 0 for a successful split,
                1 to signal a leaf creation.
        """

        # parameters
        cdef int min_samples_leaf = self.min_samples_leaf
        cdef double lmbda = self.lmbda

        cdef double* gini_indices = NULL
        cdef double* distribution = NULL
        cdef int* valid_features = NULL
        cdef int* valid_indices = NULL
        cdef int  valid_features_count = 0
        cdef int  chosen_ndx
        cdef int  chosen_feature_ndx

        cdef int i
        cdef int j
        cdef int k

        cdef int result = 0

        if node.pos_count > 0 and node.pos_count < node.count:

            gini_indices = <double *>malloc(node.features_count * sizeof(double))
            distribution = <double *>malloc(node.features_count * sizeof(double))
            valid_features = <int *>malloc(node.features_count * sizeof(int))
            valid_indices = <int *>malloc(node.features_count * sizeof(int))

            for j in range(node.features_count):

                # validate split
                if node.left_counts[j] >= min_samples_leaf and node.right_counts[j] >= min_samples_leaf:
                    valid_indices[valid_features_count] = j
                    valid_features[valid_features_count] = node.features[j]
                    gini_indices[valid_features_count] = compute_gini(node.count,
                                                                      node.left_counts[j],
                                                                      node.right_counts[j],
                                                                      node.left_pos_counts[j],
                                                                      node.right_pos_counts[j])
                    valid_features_count += 1

            if valid_features_count > 0:

                # remove invalid features
                gini_indices = <double *>realloc(gini_indices, valid_features_count * sizeof(double))
                distribution = <double *>realloc(distribution, valid_features_count * sizeof(double))
                valid_features = <int *>realloc(valid_features, valid_features_count * sizeof(int))
                valid_indices = <int *>realloc(valid_indices, valid_features_count * sizeof(int))

                # generate and sample from the distribution
                generate_distribution(lmbda, distribution, gini_indices, valid_features_count)
                chosen_ndx = sample_distribution(distribution, valid_features_count)
                chosen_feature = valid_features[chosen_ndx]
                chosen_feature_ndx = valid_indices[chosen_ndx]

                # assign results from chosen feature
                split.left_indices = <int *>malloc(node.left_counts[chosen_feature_ndx] * sizeof(int))
                split.right_indices = <int *>malloc(node.right_counts[chosen_feature_ndx] * sizeof(int))
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
                split.p = parent_p * distribution[chosen_ndx]

                # remove chosen feature from descendent nodes
                split.features_count = node.features_count - 1
                split.features = <int *>malloc(split.features_count * sizeof(int))
                j = 0
                for i in range(node.features_count):
                    if node.features[i] != split.feature:
                        split.features[j] = node.features[i]
                        j += 1

            else:
                result = 1

            free(gini_indices)
            free(distribution)
            free(valid_features)
            free(valid_indices)

        else:
            result = 1

        return result


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int compute_splits(self, Node** node_ptr, int** X, int* y,
                            int* samples, int n_samples,
                            int* features, int n_features) nogil:
        """
        Update the metadata of this node.
        """
        cdef Node* node = node_ptr[0]

        cdef int count = n_samples
        cdef int pos_count = 0

        cdef int* left_counts = <int *>malloc(n_features * sizeof(int))
        cdef int* left_pos_counts = <int *>malloc(n_features * sizeof(int))
        cdef int* right_counts = <int *>malloc(n_features * sizeof(int))
        cdef int* right_pos_counts = <int *>malloc(n_features * sizeof(int))

        cdef int left_count
        cdef int left_pos_count

        cdef int i

        # count number of pos labels
        for i in range(n_samples):
            if y[samples[i]] == 1:
                pos_count += 1

        # compute statistics for each attribute
        for j in range(n_features):

            left_count = 0
            left_pos_count = 0

            for i in range(n_samples):

                if X[samples[i]][features[j]] == 1:
                    left_count += 1
                    left_pos_count += y[samples[i]]

            left_counts[j] = left_count
            left_pos_counts[j] = left_pos_count
            right_counts[j] = count - left_count
            right_pos_counts[j] = pos_count - left_pos_count

        node.count = count
        node.pos_count = pos_count
        node.features_count = n_features
        node.features = features
        node.left_counts = left_counts
        node.left_pos_counts = left_pos_counts
        node.right_counts = right_counts
        node.right_pos_counts = right_pos_counts

