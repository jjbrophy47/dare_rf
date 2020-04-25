
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdlib cimport free
from libc.stdlib cimport rand
from libc.stdlib cimport srand
from libc.stdlib cimport RAND_MAX
from libc.stdio cimport printf
from libc.math cimport exp

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

# constants
cdef inline UINT32_t DEFAULT_SEED = 1

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
cdef double compute_gini(double count, double left_count, double right_count,
                         int left_pos_count, int right_pos_count) nogil:
    """
    Compute the Gini index of this attribute.
    """
    cdef double weight
    cdef double pos_prob
    cdef double neg_prob

    cdef double index
    cdef double left_weighted_index = 0
    cdef double right_weighted_index = 0

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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int generate_distribution(double lmbda, double* distribution,
                               double* gini_indices, int n_gini_indices) nogil:
    """
    Generate a probability distribution based on the Gini index values.
    """
    cdef int i
    cdef double normalizing_constant = 0

    cdef double min_gini = 1
    cdef int first_min = -1

    # find min Gini
    for i in range(n_gini_indices):
        if gini_indices[i] < min_gini:
            first_min = i
            min_gini = gini_indices[i]

    # determine if tree is in deterministic mode
    if lmbda < 0 or exp(- lmbda * min_gini) == 0:
        for i in range(n_gini_indices):
            distribution[i] = 0
        distribution[first_min] = 1

    # generate probability distribution over the features
    else:
        for i in range(n_gini_indices):
            distribution[i] = exp(- lmbda * gini_indices[i])
            normalizing_constant += distribution[i]

        for i in range(n_gini_indices):
            distribution[i] /= normalizing_constant

    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int sample_distribution(double* distribution, int n_distribution,
                             UINT32_t* random_state) nogil:
    """
    Randomly sample a feature from the probability distribution.
    """
    cdef int i = -1
    cdef double weight = 0

    weight = rand_uniform(0, 1, random_state)

    for i in range(n_distribution):
        if weight < distribution[i]:
            break
        weight -= distribution[i]

    return i

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int* convert_int_ndarray(np.ndarray arr):
    """
    Converts a numpy array into a C int array.
    """
    cdef int n_elem = arr.shape[0]
    cdef int* new_arr = <int *>malloc(n_elem * sizeof(int))

    for i in range(n_elem):
        new_arr[i] = arr[i]

    return new_arr

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int* copy_int_array(int* arr, int n_elem) nogil:
    """
    Copies a C int array into a new C int array.
    """
    cdef int* new_arr = <int *>malloc(n_elem * sizeof(int))

    for i in range(n_elem):
        new_arr[i] = arr[i]

    return new_arr

cdef void dealloc(Node *node) nogil:
    """
    Recursively free all nodes in the subtree.
    """
    if not node:
        return

    dealloc(node.left)
    dealloc(node.right)

    # free contents of the node
    if node.features:
        free(node.features)
        free(node.left_counts)
        free(node.left_pos_counts)
        free(node.right_counts)
        free(node.right_pos_counts)

    if node.is_leaf:
        free(node.leaf_samples)

    else:
        free(node.left)
        free(node.right)
