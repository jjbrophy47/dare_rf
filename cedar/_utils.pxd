import numpy as np
cimport numpy as np

from ._tree cimport Node

cdef double get_random() nogil
cdef double compute_gini(double count, double left_count, double right_count,
                         int left_pos_count, int right_pos_count) nogil
cdef int generate_distribution(double lmbda, double* distribution,
                               double* gini_indices, int n_gini_indices) nogil
cdef int sample_distribution(double* distribution, int n_distribution) nogil
cdef int* convert_int_ndarray(np.ndarray arr)
cdef int* copy_int_array(int* arr, int n_elem) nogil
cdef void set_srand(int random_state) nogil
cdef void dealloc(Node *node) nogil