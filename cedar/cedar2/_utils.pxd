import numpy as np
cimport numpy as np

ctypedef np.npy_uint32 UINT32_t

from ._tree cimport Node

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


# random utility methods
cdef UINT32_t our_rand_r(UINT32_t* seed) nogil

cdef double rand_uniform(double low, double high, UINT32_t* random_state) nogil

# split score utility methods
cdef double compute_split_score(bint use_gini, double count, double left_count,
                                double right_count, int left_pos_count,
                                int right_pos_count) nogil

cdef double compute_gini(double count, double left_count, double right_count,
                         int left_pos_count, int right_pos_count) nogil

cdef double compute_entropy(double count, double left_count, double right_count,
                            int left_pos_count, int right_pos_count) nogil

# SSPD methods
cdef double find_max_divergence(double* sspd1, double* sspd2,
                                int n_features) nogil

cdef int generate_distribution(double lmbda, double** distribution_ptr,
                               double* scores, int n_scores,
                               int n_samples) nogil

cdef int sample_distribution(double* distribution, int n_distribution,
                             UINT32_t* random_state) nogil

# helper methods
cdef int* convert_int_ndarray(np.ndarray arr)

cdef int* copy_int_array(int* arr, int n_elem) nogil

cdef void dealloc(Node *node) nogil
