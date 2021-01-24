# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
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

from ._tree cimport UNDEF
from ._tree cimport UNDEF_LEAF_VAL

# constants
cdef inline UINT32_t DEFAULT_SEED = 1
cdef double MAX_DBL = 1.79768e+308

# SAMPLING METHODS

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


cdef inline SIZE_t rand_int(SIZE_t low, SIZE_t high,
                            UINT32_t* random_state) nogil:
    """
    Generate a random integer in [low; end).
    """
    return low + our_rand_r(random_state) % (high - low)


cdef inline double rand_uniform(double low, double high,
                                UINT32_t* random_state) nogil:
    """
    Generate a random double in [low; high).
    """
    return ((high - low) * <double> our_rand_r(random_state) /
            <double> RAND_R_MAX) + low

# SCORING METHODS

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


# FEATURE / THRESHOLD METHODS


cdef Feature* create_feature(SIZE_t feature_index) nogil:
    """
    Allocate memory for a feature object.
    """
    cdef Feature* feature = <Feature *>malloc(sizeof(Feature))
    feature.index = feature_index
    feature.thresholds = NULL
    feature.n_thresholds = 0
    return feature


cdef Threshold* create_threshold(DTYPE_t value,
                                 SIZE_t  n_left_samples,
                                 SIZE_t  n_right_samples) nogil:
    """
    Allocate memory for a threshold object.
    """
    cdef Threshold* threshold = <Threshold *>malloc(sizeof(Threshold))
    threshold.value = value
    threshold.n_left_samples = n_left_samples
    threshold.n_right_samples = n_right_samples
    threshold.v1 = 0
    threshold.v2 = 0
    threshold.n_v1_samples = 0
    threshold.n_v1_pos_samples = 0
    threshold.n_v2_samples = 0
    threshold.n_v2_pos_samples = 0
    threshold.n_left_pos_samples = 0
    threshold.n_right_pos_samples = 0
    return threshold


cdef Feature* copy_feature(Feature* feature) nogil:
    """
    Copies the contents of a feature to a new feature.
    """
    cdef Feature* f2 = <Feature *>malloc(sizeof(Feature))
    f2.index = feature.index
    f2.n_thresholds = feature.n_thresholds
    f2.thresholds = <Threshold **>malloc(feature.n_thresholds * sizeof(Threshold *))
    for k in range(feature.n_thresholds):
        f2.thresholds[k] = copy_threshold(feature.thresholds[k])
    return f2


cdef Threshold* copy_threshold(Threshold* threshold) nogil:
    """
    Copies the contents of a threshold to a new threshold.
    """
    cdef Threshold* t2 = <Threshold *>malloc(sizeof(Threshold))

    t2.v1 = threshold.v1
    t2.v2 = threshold.v2
    t2.value = threshold.value
    t2.n_v1_samples = threshold.n_v1_samples
    t2.n_v1_pos_samples = threshold.n_v1_pos_samples
    t2.n_v2_samples = threshold.n_v2_samples
    t2.n_v2_pos_samples = threshold.n_v2_pos_samples
    t2.n_left_samples = threshold.n_left_samples
    t2.n_left_pos_samples = threshold.n_left_pos_samples
    t2.n_right_samples = threshold.n_right_samples
    t2.n_right_pos_samples = threshold.n_right_pos_samples

    return t2


cdef void free_features(Feature** features,
                           SIZE_t n_features) nogil:
    """
    Deallocate a features array and all thresholds.
    """
    cdef SIZE_t j = 0

    # free each feature and then the array
    if features != NULL:
        for j in range(n_features):
            free_feature(features[j])
        free(features)


cdef void free_feature(Feature* feature) nogil:
    """
    Frees all properties of this feature, and then the feature.
    """
    if feature != NULL:
        if feature.thresholds != NULL:
            free_thresholds(feature.thresholds, feature.n_thresholds)
        free(feature)


cdef void free_thresholds(Threshold** thresholds,
                          SIZE_t n_thresholds) nogil:
    """
    Deallocate a thresholds array and its contents
    """
    cdef SIZE_t k = 0

    # free each threshold and then the array
    if thresholds != NULL:
        for k in range(n_thresholds):
            free(thresholds[k])
        free(thresholds)


# INTLIST METHODS


cdef IntList* create_intlist(SIZE_t n_elem, bint initialize) nogil:
    """
    Allocate memory for:
    -IntList object.
    -IntList.arr object with size n_elem.
    If `initialize` is True, Set IntList.n = n, IntList.n = 0.
    """
    cdef IntList* obj = <IntList *>malloc(sizeof(IntList))
    obj.arr = <SIZE_t *>malloc(n_elem * sizeof(SIZE_t))

    # set n
    if initialize:
        obj.n = n_elem
    else:
        obj.n = 0

    return obj


cdef IntList* copy_intlist(IntList* obj, SIZE_t n_elem) nogil:
    """
    -Creates a new IntList object.
    -Allocates the `arr` with size `n_elem`.
    -Copies values from `obj.arr` up to `obj.n`.
    -`n` is set to `obj.n`.

    NOTE: n_elem >= obj.n.
    """
    cdef IntList* new_obj = create_intlist(n_elem, 0)

    # copy array values
    for i in range(obj.n):
        new_obj.arr[i] = obj.arr[i]

    # set n
    new_obj.n = obj.n

    return new_obj


cdef void free_intlist(IntList* obj) nogil:
    """
    Deallocate IntList object.
    """
    free(obj.arr)
    free(obj)
    obj = NULL


# ARRAY METHODS


cdef SIZE_t* convert_int_ndarray(np.ndarray arr):
    """
    Converts a numpy array into a C int array.
    """
    cdef SIZE_t  n_elem = arr.shape[0]
    cdef SIZE_t* new_arr = <SIZE_t *>malloc(n_elem * sizeof(SIZE_t))

    for i in range(n_elem):
        new_arr[i] = arr[i]

    return new_arr


cdef INT32_t* copy_int_array(INT32_t* arr, SIZE_t n_elem) nogil:
    """
    Copies a C int array into a new C int array.
    """
    cdef INT32_t* new_arr = <INT32_t *>malloc(n_elem * sizeof(INT32_t))

    for i in range(n_elem):
        new_arr[i] = arr[i]

    return new_arr


cdef SIZE_t* copy_indices(SIZE_t* arr, SIZE_t n_elem) nogil:
    """
    Copies a C int array into a new C int array.
    """
    cdef SIZE_t* new_arr = <SIZE_t *>malloc(n_elem * sizeof(SIZE_t))

    for i in range(n_elem):
        new_arr[i] = arr[i]

    return new_arr


# NODE METHODS


cdef void split_samples(Node*        node,
                        DTYPE_t**    X,
                        INT32_t*     y,
                        IntList*     samples,
                        SplitRecord* split,
                        bint         copy_constant_features) nogil:
    """
    Split samples based on the chosen feature / threshold.

    NOTE: frees `samples.arr` and `samples` object.
    """

    # split samples based on the chosen feature / threshold
    split.left_samples = create_intlist(samples.n, 0)
    split.right_samples = create_intlist(samples.n, 0)

    # loop through the deleted samples
    for i in range(samples.n):

        # add sample to the left branch
        if X[samples.arr[i]][node.chosen_feature.index] <= node.chosen_threshold.value:
            split.left_samples.arr[split.left_samples.n] = samples.arr[i]
            split.left_samples.n += 1

        # add sample to the right branch
        else:
            split.right_samples.arr[split.right_samples.n] = samples.arr[i]
            split.right_samples.n += 1

    # assign left branch deleted samples
    if split.left_samples.n > 0:
        split.left_samples.arr = <SIZE_t *>realloc(split.left_samples.arr,
                                                   split.left_samples.n * sizeof(SIZE_t))
    else:
        split.left_samples.n = 0
        free_intlist(split.left_samples)

    # assign right branch deleted samples
    if split.right_samples.n > 0:
        split.right_samples.arr = <SIZE_t *>realloc(split.right_samples.arr,
                                                    split.right_samples.n * sizeof(SIZE_t))
    else:
        split.right_samples.n = 0
        free_intlist(split.right_samples)

    # copy constant features array for both branches
    if copy_constant_features:
        split.left_constant_features = copy_intlist(node.constant_features, node.constant_features.n)
        split.right_constant_features = copy_intlist(node.constant_features, node.constant_features.n)

    # clean up, no more use for the original samples array
    free_intlist(samples)


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

        # clear chosen feature
        if node.chosen_feature != NULL:
            if node.chosen_feature.thresholds != NULL:
                for k in range(node.chosen_feature.n_thresholds):
                    free(node.chosen_feature.thresholds[k])
                free(node.chosen_feature.thresholds)
            free(node.chosen_feature)

        # clear chosen threshold
        if node.chosen_threshold != NULL:
            free(node.chosen_threshold)

        # clear constant features
        if node.constant_features != NULL:
            free_intlist(node.constant_features)

        # clear features array
        if node.features != NULL:
            for j in range(node.n_features):

                if node.features[j] != NULL:
                    for k in range(node.features[j].n_thresholds):
                        free(node.features[j].thresholds[k])

                    free(node.features[j].thresholds)
                    free(node.features[j])

            free(node.features)

        # free children
        free(node.left)
        free(node.right)

    # reset general node properties
    node.left = NULL
    node.right = NULL

    # reset leaf properties
    node.is_leaf = False
    node.value = UNDEF_LEAF_VAL
    node.leaf_samples = NULL

    # reset decision node properties
    node.features = NULL
    node.n_features = 0
    node.constant_features = NULL
    node.chosen_feature = NULL
    node.chosen_threshold = NULL
