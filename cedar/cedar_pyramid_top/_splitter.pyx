
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport compute_split_score
from ._utils cimport generate_distribution
from ._utils cimport sample_distribution
from ._utils cimport RAND_R_MAX

cdef class _Splitter:
    """
    Splitter class.
    Finds the best splits on dense data, one split at a time.
    """

    def __cinit__(self, int min_samples_leaf, double lmbda, bint use_gini,
                  object random_state):
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
        use_gini : bool
            If True, use the Gini index splitting criterion; otherwise
            use entropy.
        random_state : object
            The user inputted random state to be used for pseudo-randomness
        """
        self.min_samples_leaf = min_samples_leaf
        self.lmbda = lmbda
        self.use_gini = use_gini
        self.random_state = random_state
        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int split_node(self, Node* node, int** X, int* y,
                        int* samples, int n_samples,
                        int topd, double atol, SplitRecord *split) nogil:
        """
        Splits the node by sampling from the valid feature distribution.
        Returns 0 for a successful split,
                1 to signal a leaf creation.
        """

        # parameters
        cdef int min_samples_leaf = self.min_samples_leaf
        cdef double lmbda = self.lmbda
        cdef bint use_gini = self.use_gini

        cdef double* split_scores = NULL
        cdef double* distribution = NULL
        cdef int  chosen_ndx

        cdef UINT32_t* random_state = &self.rand_r_state

        cdef int i
        cdef int j
        cdef int k

        cdef int result = 0

        if node.pos_count > 0 and node.pos_count < node.count:

            split_scores = <double *>malloc(node.features_count * sizeof(double))
            distribution = <double *>malloc(node.features_count * sizeof(double))

            for j in range(node.features_count):
                split_scores[j] = compute_split_score(use_gini, node.count, node.left_counts[j],
                                                      node.right_counts[j], node.left_pos_counts[j],
                                                      node.right_pos_counts[j])

            # generate and sample from the distribution
            if node.depth >= topd:
                lmbda = 0

            generate_distribution(lmbda, &distribution, split_scores,
                                  node.features_count, n_samples, use_gini, atol)
            chosen_ndx = sample_distribution(distribution, node.features_count, random_state)
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
            split.sspd = distribution

            # remove chosen feature from descendent nodes
            split.features_count = node.features_count - 1
            split.left_features = <int *>malloc(split.features_count * sizeof(int))
            split.right_features = <int *>malloc(split.features_count * sizeof(int))
            j = 0
            for i in range(node.features_count):
                if node.features[i] != split.feature:
                    split.left_features[j] = node.features[i]
                    split.right_features[j] = node.features[i]
                    j += 1

            free(split_scores)

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
