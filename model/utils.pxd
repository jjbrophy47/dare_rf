import numpy as np
cimport numpy as np

from tree cimport UINT32_t

# cdef enum:
#     # Max value for our rand_r replacement (near the bottom).
#     # We don't use RAND_MAX because it's different across platforms and
#     # particularly tiny on Windows/MSVC.
#     RAND_R_MAX = 0x7FFFFFFF

# ctypedef fused realloc_ptr:
#     # Add pointer types here as needed.
#     (DTYPE_t*)
#     (SIZE_t*)
#     (unsigned char*)
#     (DOUBLE_t*)
#     (DOUBLE_t**)
#     (Node*)
#     (Node**)
#     (StackRecord*)

# cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *

# cdef np.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size)

cdef UINT32_t our_rand_r(UINT32_t* seed) nogil

# cdef inline int rand_int(int low, int high, UINT32_t* random_state) nogil

cdef double rand_uniform(double low, double high, UINT32_t* random_state) nogil
cdef double get_random(int random_state) nogil

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
