import numpy as np
cimport numpy as np

cdef double get_random() nogil
cdef double compute_gini(double count, double left_count, double right_count,
                         int left_pos_count, int right_pos_count) nogil
cdef int generate_distribution(double lmbda, double* distribution,
                               double* gini_indices, int n_gini_indices) nogil
cdef int sample_distribution(double* distribution, int n_distribution) nogil
cdef int* convert_int_ndarray(np.ndarray arr)

# =============================================================================
# Stack data structure
# =============================================================================

# A record on the stack for depth-first tree growing
cdef struct StackRecord:
    int depth
    int parent
    double parent_p
    bint is_left
    int* samples
    int n_samples
    int* features
    int n_features

cdef class Stack:
    cdef int capacity
    cdef int top
    cdef StackRecord* stack_

    cdef bint is_empty(self) nogil
    cdef int push(self, int depth, int parent, double parent_p, bint is_left,
                  int* samples, int n_samples, int* features, int n_features) nogil
    cdef int pop(self, StackRecord* res) nogil

# =============================================================================
# Removal stack structure
# =============================================================================

# A record on the stack for depth-first tree growing
cdef struct RemovalStackRecord:
    int depth
    int node_id
    bint is_left
    int parent
    double parent_p
    int* samples
    int n_samples

cdef class RemovalStack:
    cdef int capacity
    cdef int top
    cdef RemovalStackRecord* stack_

    cdef bint is_empty(self) nogil
    cdef int push(self, int depth, int node_id, bint is_left, int parent,
                  double parent_p, int* samples, int n_samples) nogil
    cdef int pop(self, RemovalStackRecord* res) nogil

# =============================================================================
# Basic Int stack
# =============================================================================

cdef class IntStack:
    cdef int capacity
    cdef int top
    cdef int* stack_

    cdef bint is_empty(self) nogil
    cdef int push(self, int node_id) nogil
    cdef int pop(self) nogil
