
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf
from libc.math cimport exp
from libc.math cimport isnan

from libc.stdlib cimport rand
from libc.stdlib cimport srand
from libc.stdlib cimport RAND_MAX

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport get_random

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

    def __dealloc__(self):
        """Destructor."""
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int node_split(self, int[::1, :] X, int[::1] y, int[::1] f,
                        int* samples, int* features, int n_features,
                        SplitRecord* split, Meta* meta):
        """
        Find the best split in the node data.
        This is a placeholder method. The majority of computation will be done here.
        It should return -1 upon errors.
        """

        cdef int min_samples_leaf = self.min_samples_leaf

        cdef int i
        cdef int j
        cdef int k
        cdef int chosen_ndx
        cdef int chosen_feature

        cdef int n_samples = meta.count
        cdef int count = meta.count
        cdef int pos_count = 0
        cdef int left_count
        cdef int left_pos_count
        cdef int right_count
        cdef int right_pos_count

        cdef int feature_count = 0
        cdef int result = 0

        cdef double* gini_indices = <double *>malloc(n_features * sizeof(double))
        cdef double* distribution = <double *>malloc(n_features * sizeof(double))
        cdef int* valid_features = <int *>malloc(n_features * sizeof(int))

        cdef int* left_counts = <int *>malloc(n_features * sizeof(int))
        cdef int* left_pos_counts = <int *>malloc(n_features * sizeof(int))
        cdef int* right_counts = <int *>malloc(n_features * sizeof(int))
        cdef int* right_pos_counts = <int *>malloc(n_features * sizeof(int))

        # count number of pos labels
        for i in range(n_samples):
            if y[samples[i]] == 1:
                pos_count += 1

        # compute statistics for each attribute
        for j in range(n_features):

            left_count = 0
            left_pos_count = 0

            for i in range(n_samples):

                if X[samples[i], features[j]] == 1:
                    left_count += 1
                    left_pos_count += y[samples[i]]

            right_count = count - left_count
            right_pos_count = pos_count - left_pos_count

            # validate split
            if left_count >= min_samples_leaf and right_count >= min_samples_leaf:
                valid_features[feature_count] = features[j]
                gini_indices[feature_count] = self._compute_gini(count, left_count, right_count,
                                                                 left_pos_count, right_pos_count)
                # printf('gini_indices[%d]: %.7f\n', feature_count, gini_indices[feature_count])

                # save metadata
                left_counts[feature_count] = left_count
                left_pos_counts[feature_count] = left_pos_count
                right_counts[feature_count] = right_count
                right_pos_counts[feature_count] = right_pos_count

                feature_count += 1

        if feature_count > 0:

            # remove invalid features
            gini_indices = <double *>realloc(gini_indices, feature_count * sizeof(double))
            distribution = <double *>realloc(distribution, feature_count * sizeof(double))
            valid_features = <int *>realloc(valid_features, feature_count * sizeof(int))

            left_counts = <int *>realloc(left_counts, feature_count * sizeof(int))
            left_pos_counts = <int *>realloc(left_pos_counts, feature_count * sizeof(int))
            right_counts = <int *>realloc(right_counts, feature_count * sizeof(int))
            right_pos_counts = <int *>realloc(right_pos_counts, feature_count * sizeof(int))

            # generate and sample from the distribution
            self._generate_distribution(distribution, gini_indices, feature_count)
            chosen_ndx = self._sample_distribution(distribution, feature_count)

            # assign results from chosen feature
            split.left_indices = <int *>malloc(left_counts[chosen_ndx] * sizeof(int))
            split.right_indices = <int *>malloc(right_counts[chosen_ndx] * sizeof(int))
            j = 0
            k = 0
            for i in range(n_samples):
                if X[samples[i], valid_features[chosen_ndx]] == 1:
                    split.left_indices[j] = samples[i]
                    j += 1
                else:
                    split.right_indices[k] = samples[i]
                    k += 1
            split.left_count = j
            split.right_count = k
            split.feature = valid_features[chosen_ndx]
            split.features = valid_features
            split.n_features = feature_count

            meta.pos_count = pos_count
            meta.feature_count = feature_count
            meta.left_counts = left_counts
            meta.left_pos_counts = left_pos_counts
            meta.right_counts = right_counts
            meta.right_pos_counts = right_pos_counts
            meta.features = valid_features

        else:
            result = -2

        # clean up
        free(gini_indices)
        free(distribution)

        return result

    @cython.cdivision(True)
    cdef double _compute_gini(self, double count, double left_count, double right_count, 
                              int left_pos_count, int right_pos_count) nogil:
        """
        Compute the Gini index of this attribute.
        """
        cdef double weight
        cdef double pos_prob
        cdef double neg_prob

        cdef double index
        cdef double left_weighted_index
        cdef double right_weighted_index

        weight = left_count / count
        pos_prob = left_pos_count / left_count
        neg_prob = 1 - pos_prob
        index = 1 - (pos_prob * pos_prob) - (neg_prob * neg_prob)
        left_weighted_index = weight * index

        weight = right_count / count
        pos_prob = right_pos_count / right_count
        neg_prob = 1 - pos_prob
        index = 1 - (pos_prob * pos_prob) - (neg_prob * neg_prob)
        right_weighted_index = weight * index

        return left_weighted_index + right_weighted_index

    @cython.cdivision(True)
    cdef int _generate_distribution(self, double* distribution, double* gini_indices,
                                    int n_gini_indices) nogil:
        """
        Generate a probability distribution based on the Gini index values.
        """
        cdef int i
        cdef double lmbda = self.lmbda
        cdef double normalizing_constant = 0

        cdef double min_gini = 1
        cdef int n_min = 0
        cdef int first_min = -1

        # find min and max Gini values
        for i in range(n_gini_indices):
            if gini_indices[i] < min_gini:
                n_min = 1
                first_min = i
                min_gini = gini_indices[i]
            elif gini_indices[i] == min_gini:
                n_min += 1

        # lambda too high, go into deterministic mode
        if exp(- lmbda * min_gini / 5) == 0:
            for i in range(n_gini_indices):
                distribution[i] = 0
            distribution[first_min] = 1
            normalizing_constant = 1

        # generate probability distribution over the features
        else:
            for i in range(n_gini_indices):
                distribution[i] = exp(- lmbda * gini_indices[i] / 5)
                normalizing_constant += distribution[i]

            for i in range(n_gini_indices):
                distribution[i] /= normalizing_constant
                # printf('distribution[%d]: %.7f\n', i, distribution[i])

        return 0

    cdef int _sample_distribution(self, double* distribution, int n_distribution) nogil:
        """
        Randomly sample a feature from the probability distribution.
        """
        cdef int i
        cdef double weight = 0

        weight = get_random()
        # printf('initial weight: %.7f\n', weight)

        for i in range(n_distribution):
            if weight < distribution[i]:
                break
            weight -= distribution[i]

        return i
