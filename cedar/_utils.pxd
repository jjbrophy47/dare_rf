import numpy as np
cimport numpy as np

cdef double get_random() nogil
cdef double _compute_gini(double count, double left_count, double right_count,
                          int left_pos_count, int right_pos_count) nogil
cdef int _generate_distribution(double lmbda, double* distribution,
                                double* gini_indices, int n_gini_indices) nogil
cdef int _sample_distribution(self, double* distribution, int n_distribution) nogil

# =============================================================================
# Stack data structure
# =============================================================================

# A record on the stack for depth-first tree growing
cdef struct StackRecord:
    int depth
    int parent
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
    cdef int push(self, int depth, int parent, bint is_left, int* samples,
                  int n_samples, int* features, int n_features) nogil
    cdef int pop(self, StackRecord* res) nogil

# =============================================================================
# Removal stack structure
# =============================================================================

# A record on the stack for depth-first tree growing
cdef struct RemovalStackRecord:
    int depth
    int node_id
    double parent_p
    int* samples
    int* remove_samples
    int n_samples

cdef class RemovalStack:
    cdef int capacity
    cdef int top
    cdef StackRecord* stack_

    cdef bint is_empty(self) nogil
    cdef int push(self, int depth, int node_id, double parent_p,
                  int* samples, int n_samples) nogil
    cdef int pop(self, StackRecord* res) nogil
