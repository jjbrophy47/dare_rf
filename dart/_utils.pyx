"""
Utility methods.
"""
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdlib cimport free
from libc.stdlib cimport rand
from libc.stdlib cimport srand
from libc.stdlib cimport RAND_MAX
from libc.stdio cimport printf
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport log2
from libc.math cimport fabs

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

# constants
cdef inline UINT32_t DEFAULT_SEED = 1
cdef double MAX_DBL = 1.79768e+308

cdef inline double rand_uniform(double low, double high,
                                UINT32_t* random_state) nogil:
    """
    Generate a random double in [low; high).
    """
    return ((high - low) * <double> our_rand_r(random_state) /
            <double> RAND_R_MAX) + low

# rand_r replacement using a 32bit XorShift generator
# See http://www.jstatsoft.org/v08/i14/paper for details
cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    """
    Generate a pseudo-random np.uint32 from a np.uint32 seed.
    """
    # seed shouldn't ever be 0.
    if (seed[0] == 0): seed[0] = DEFAULT_SEED

    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    # Note: we must be careful with the final line cast to np.uint32 so that
    # the function behaves consistently across platforms.
    #
    # The following cast might yield different results on different platforms:
    # wrong_cast = <UINT32_t> RAND_R_MAX + 1
    #
    # We can use:
    # good_cast = <UINT32_t>(RAND_R_MAX + 1)
    # or:
    # cdef np.uint32_t another_good_cast = <UINT32_t>RAND_R_MAX + 1
    return seed[0] % <UINT32_t>(RAND_R_MAX + 1)


@cython.cdivision(True)
cdef DTYPE_t compute_split_score(bint    use_gini,
                                 DTYPE_t count,
                                 DTYPE_t left_count,
                                 DTYPE_t right_count,
                                 SIZE_t  left_pos_count,
                                 SIZE_t  right_pos_count) nogil:
    """
    Computes either the Gini index or entropy given this attribute.
    """
    cdef DTYPE_t result

    if use_gini:
        result = compute_gini(count, left_count, right_count,
                              left_pos_count, right_pos_count)

    else:
        result = compute_entropy(count, left_count, right_count,
                                 left_pos_count, right_pos_count)

    return result



@cython.cdivision(True)
cdef DTYPE_t compute_gini(DTYPE_t count,
                          DTYPE_t left_count,
                          DTYPE_t right_count,
                          SIZE_t  left_pos_count,
                          SIZE_t  right_pos_count) nogil:
    """
    Compute the Gini index given this attribute.
    """
    cdef DTYPE_t weight
    cdef DTYPE_t pos_prob
    cdef DTYPE_t neg_prob

    cdef DTYPE_t index
    cdef DTYPE_t left_weighted_index = 0
    cdef DTYPE_t right_weighted_index = 0

    if left_count > 0:
        weight = left_count / count
        pos_prob = left_pos_count / left_count
        neg_prob = 1 - pos_prob
        index = 1 - (pos_prob * pos_prob) - (neg_prob * neg_prob)
        left_weighted_index = weight * index

    if right_count > 0:
        weight = right_count / count
        pos_prob = right_pos_count / right_count
        neg_prob = 1 - pos_prob
        index = 1 - (pos_prob * pos_prob) - (neg_prob * neg_prob)
        right_weighted_index = weight * index

    return left_weighted_index + right_weighted_index

@cython.cdivision(True)
cdef DTYPE_t compute_entropy(DTYPE_t count,
                             DTYPE_t left_count,
                             DTYPE_t right_count,
                             SIZE_t  left_pos_count,
                             SIZE_t  right_pos_count) nogil:
    """
    Compute the mutual information given this attribute.
    """
    cdef DTYPE_t weight
    cdef DTYPE_t pos_prob
    cdef DTYPE_t neg_prob

    cdef DTYPE_t entropy
    cdef DTYPE_t left_weighted_entropy = 0
    cdef DTYPE_t right_weighted_entropy = 0

    if left_count > 0:
        weight = left_count / count
        pos_prob = left_pos_count / left_count
        neg_prob = 1 - pos_prob

        entropy = 0
        if pos_prob > 0:
            entropy -= pos_prob * log2(pos_prob)
        if neg_prob > 0:
            entropy -= neg_prob * log2(neg_prob)

        left_weighted_entropy = weight * entropy

    if right_count > 0:
        weight = right_count / count
        pos_prob = right_pos_count / right_count
        neg_prob = 1 - pos_prob

        entropy = 0
        if pos_prob > 0:
            entropy -= pos_prob * log2(pos_prob)
        if neg_prob > 0:
            entropy -= neg_prob * log2(neg_prob)

        right_weighted_entropy = weight * entropy

    return left_weighted_entropy + right_weighted_entropy

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void split_samples(Node*        node,
                        DTYPE_t**    X,
                        INT32_t*     y,
                        SIZE_t*      samples,
                        SIZE_t       n_samples,
                        SplitRecord* split) nogil:
    """
    Split samples based on the chosen feature.
    """

    cdef SIZE_t i = 0
    cdef SIZE_t j = 0
    cdef SIZE_t k = 0

    # assign results from chosen feature
    split.left_samples = <SIZE_t *>malloc(n_samples * sizeof(SIZE_t))
    split.right_samples = <SIZE_t *>malloc(n_samples * sizeof(SIZE_t))
    for i in range(n_samples):
        if X[samples[i]][node.chosen_feature.index] <= node.chosen_threshold.value:
            split.left_samples[j] = samples[i]
            j += 1
        else:
            split.right_samples[k] = samples[i]
            k += 1
    split.left_samples = <SIZE_t *>realloc(split.left_samples, j * sizeof(SIZE_t))
    split.right_samples = <SIZE_t *>realloc(split.right_samples, k * sizeof(SIZE_t))
    split.n_left_samples = j
    split.n_right_samples = k

@cython.boundscheck(False)
@cython.wraparound(False)
cdef INT32_t* convert_int_ndarray(np.ndarray arr):
    """
    Converts a numpy array into a C int array.
    """
    cdef SIZE_t n_elem = arr.shape[0]
    cdef INT32_t* new_arr = <INT32_t *>malloc(n_elem * sizeof(INT32_t))

    for i in range(n_elem):
        new_arr[i] = arr[i]

    return new_arr

@cython.boundscheck(False)
@cython.wraparound(False)
cdef INT32_t* copy_int_array(INT32_t* arr, SIZE_t n_elem) nogil:
    """
    Copies a C int array into a new C int array.
    """
    cdef INT32_t* new_arr = <INT32_t *>malloc(n_elem * sizeof(INT32_t))

    for i in range(n_elem):
        new_arr[i] = arr[i]

    return new_arr

@cython.boundscheck(False)
@cython.wraparound(False)
cdef SIZE_t* copy_indices(SIZE_t* arr, SIZE_t n_elem) nogil:
    """
    Copies a C int array into a new C int array.
    """
    cdef SIZE_t* new_arr = <SIZE_t *>malloc(n_elem * sizeof(SIZE_t))

    for i in range(n_elem):
        new_arr[i] = arr[i]

    return new_arr

cdef void dealloc(Node *node) nogil:
    """
    Recursively free all nodes in the subtree.

    NOTE: Does not deallocate "root" node, that must
          be done by the caller!
    """
    if not node:
        return

    # traverse to the bottom nodes first
    dealloc(node.left)
    dealloc(node.right)

    # leaf node
    if node.is_leaf:
        free(node.leaf_samples)

    # decision node
    else:

        # loop through each feature
        for j in range(node.n_features):

            # loop through each threshold of this feature
            for k in range(node.features[j].n_thresholds):

                # free this threshold
                free(node.features[j].thresholds[k])

            # free threshold array
            free(node.features[j].thresholds)

            # free this feature
            free(node.features[j])

        # free feature array
        free(node.features)

        # free children
        free(node.left)
        free(node.right)
