# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
"""
Configuration object storing tree building parameters.
"""
cimport cython

from ._utils cimport RAND_R_MAX


cdef class _Config:
    """
    Class to store training parameters.
    """

    def __cinit__(self,
                  SIZE_t       min_samples_split,
                  SIZE_t       min_samples_leaf,
                  SIZE_t       max_depth,
                  SIZE_t       topd,
                  SIZE_t       k,
                  SIZE_t       max_features,
                  bint         use_gini,
                  object       random_state):

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.topd = topd
        self.k = k
        self.max_features = max_features
        self.use_gini = use_gini
        self.rand_r_state = random_state.randint(0, RAND_R_MAX)
