
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf
from libc.math cimport exp

from libc.stdlib cimport rand
from libc.stdlib cimport srand
from libc.stdlib cimport RAND_MAX

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from utils cimport get_random

cdef class Splitter:
    """
    Splitter class.
    Finds the best splits on dense data, one split at a time.
    """

    def __cinit__(self, int min_samples_leaf, double lmbda, UINT32_t random_state):
        """
        Parameters
        ----------
        min_samples_leaf : SIZE_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.
        """
        # self.criterion = criterion

        # self.samples = NULL
        # self.n_samples = 0
        # self.features = NULL
        # self.n_features = 0

        self.min_samples_leaf = min_samples_leaf
        self.lmbda = lmbda
        self.random_state = random_state

    def __dealloc__(self):
        """Destructor."""
        pass
        # free(self.samples)
        # free(self.features)

    cdef int init(self):
        """
        Initialize the splitter.
        Take in the input data X and the target Y.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.
        y : ndarray, dtype=int
            This is the vector of targets, or true labels, for the samples
        """
        # self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        # cdef int n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        # cdef int* samples = <int *>malloc(n_samples * sizeof(int))

        # cdef int i
        # cdef int n_features = X.shape[1]
        # cdef int* features = <int *>malloc(n_features * sizeof(int))

        # for i in range(n_features):
        #     features[i] = i
        return 0

    # cdef int node_reset(self, SIZE_t start, SIZE_t end) nogil except -1:
    #     """
    #     Reset splitter on node samples[start:end].
    #     Returns -1 in case of failure to allocate memory (and raise MemoryError)
    #     or 0 otherwise.

    #     Parameters
    #     ----------
    #     start : SIZE_t
    #         The index of the first sample to consider
    #     end : SIZE_t
    #         The index of the last sample to consider
    #     """
    #     self.start = start
    #     self.end = end
    #     self.criterion.init(self.y, self.samples, start, end)
    #     return 0

    # TODO: remove chosen attribute from leftovers if easy to do, otherwise it'll
    #       get removed at the next node.
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int node_split(self, int[::1, :] X, int[::1] y, int[::1] f,
                        int* samples, int n_samples, int* features, int n_features,
                        SplitRecord* split):
    # cpdef int node_split(self, np.ndarray[INT32_t, ndim=2] X,
    #                            np.ndarray[INT32_t, ndim=1] y):
        """
        Find the best split in the node data.
        This is a placeholder method. The majority of computation will be done here.
        It should return -1 upon errors.
        """

        # Find the best split
        # cdef int n_samples = X.shape[0]
        # cdef int n_features = X.shape[1]

        cdef int min_samples_leaf = self.min_samples_leaf

        cdef int i
        cdef int j
        cdef int k
        cdef int chosen_ndx
        cdef int chosen_feature

        cdef int count = n_samples
        cdef int pos_count = 0
        cdef int left_count
        cdef int left_pos_count
        cdef int right_count
        cdef int right_pos_count

        cdef int feature_count = 0

        # printf("heyooooo\n")

        cdef double* gini_indices = <double *>malloc(n_features * sizeof(double))
        cdef double* distribution = <double *>malloc(n_features * sizeof(double))
        cdef int* valid_features = <int *>malloc(n_features * sizeof(int))

        cdef int* left_counts = <int *>malloc(n_features * sizeof(int))
        cdef int* left_pos_counts = <int *>malloc(n_features * sizeof(int))
        cdef int* right_counts = <int *>malloc(n_features * sizeof(int))
        cdef int* right_pos_counts = <int *>malloc(n_features * sizeof(int))

        # printf("finished allocating\n")

        # count number of pos labels
        for i in range(n_samples):
            # printf("sample[%d]: %d\n", i, samples[i])
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
                    # if y[samples[i]] == 1:
                    #     left_pos_count += 1

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

        # TODO: handle feature_count of 0
        if feature_count == 0:
        # clean up
            free(gini_indices)
            free(distribution)

            free(left_counts)
            free(left_pos_counts)
            free(right_counts)
            free(right_pos_counts)
            return -2

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
        # printf('chosen feature: %d\n', valid_features[chosen_ndx])

        # printf('left count: %d\n', left_counts[chosen_ndx])
        # printf('right_count: %d\n', right_counts[chosen_ndx])

        # assign results from chosen feature
        split.left_indices = <int *>malloc(left_counts[chosen_ndx] * sizeof(int))
        split.right_indices = <int *>malloc(right_counts[chosen_ndx] * sizeof(int))
        j = 0
        k = 0
        for i in range(n_samples):
            if X[samples[i], valid_features[chosen_ndx]] == 1:
                split.left_indices[j] = samples[i]
                # printf('left_indices[%d]: %d\n', j, split.left_indices[j])
                j += 1
            else:
                split.right_indices[k] = samples[i]
                # printf('right_indices[%d]: %d\n', k, split.right_indices[k])
                k += 1
        split.left_count = j
        split.right_count = k
        split.feature = valid_features[chosen_ndx]
        split.features = valid_features
        split.n_features = feature_count

        # clean up
        free(gini_indices)
        free(distribution)

        free(left_counts)
        free(left_pos_counts)
        free(right_counts)
        free(right_pos_counts)

        return 0

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

        for i in range(n_gini_indices):
            distribution[i] = exp(- lmbda * gini_indices[i] / 5)
            normalizing_constant += distribution[i]

        for i in range(n_gini_indices):
            distribution[i] /= normalizing_constant
            # printf('distribution[%d]: %.7f\n', i, distribution[i])

        return 0

    # TODO: use random_state in get_random() function
    cdef int _sample_distribution(self, double* distribution, int n_distribution) nogil:
        """
        Randomly sample a feature from the probability distribution.
        """
        cdef int i
        cdef double weight = 0

        weight = get_random(self.random_state)
        # printf('initial weight: %.7f\n', weight)

        for i in range(n_distribution):
            if weight < distribution[i]:
                break
            weight -= distribution[i]

        return i
