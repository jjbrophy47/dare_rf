import numpy as np
cimport numpy as np

cdef double get_random() nogil

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
                  int n_samples, int* features, int n_features) nogil except -1
    cdef int pop(self, StackRecord* res) nogil
