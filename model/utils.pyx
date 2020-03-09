
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdlib cimport free
from libc.stdlib cimport rand
from libc.stdlib cimport srand
from libc.stdlib cimport RAND_MAX
from libc.time cimport time

import numpy as np
cimport numpy as np

cdef inline UINT32_t DEFAULT_SEED = 1

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

# cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *:
#     # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
#     # 0.20.1 to crash.
#     cdef size_t nbytes = nelems * sizeof(p[0][0])
#     if nbytes / sizeof(p[0][0]) != nelems:
#         # Overflow in the multiplication
#         with gil:
#             raise MemoryError("could not allocate (%d * %d) bytes"
#                               % (nelems, sizeof(p[0][0])))
#     cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
#     if tmp == NULL:
#         with gil:
#             raise MemoryError("could not allocate %d bytes" % nbytes)

#     p[0] = tmp
#     return tmp  # for convenience

# cdef inline np.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size):
#     """
#     Return copied data as 1D numpy array of intp's.
#     """
#     cdef np.npy_intp shape[1]
#     shape[0] = <np.npy_intp> size
#     return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INTP, data).copy()

# cpdef check_random_state(seed):
#     """Turn seed into a np.random.RandomState instance
#     Parameters
#     ----------
#     seed : None | int | instance of RandomState
#         If seed is None, return the RandomState singleton used by np.random.
#         If seed is an int, return a new RandomState instance seeded with seed.
#         If seed is already a RandomState instance, return it.
#         Otherwise raise ValueError.
#     """
#     if seed is None or seed is np.random:
#         return np.random.mtrand._rand
#     if isinstance(seed, numbers.Integral):
#         return np.random.RandomState(seed)
#     if isinstance(seed, np.random.RandomState):
#         return seed
#     raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     # ' instance' % seed)

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

# cdef inline int rand_int(int low, int high, UINT32_t* random_state) nogil:
#     """
#     Generate a random integer in [low; end).
#     """
#     return low + our_rand_r(random_state) % (high - low)

# cdef np.ndarray _get_double_ndarray(self, double *data):
#     """
#     Wraps array as a 1-d NumPy array.
#     The array keeps a reference to this Tree, which manages the underlying memory.
#     """
#     cdef np.npy_intp shape[1]
#     shape[0] = self.node_count
#     cdef np.ndarray arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, data)
#     Py_INCREF(self)
#     arr.base = <PyObject*> self
#     return arr

cdef inline double rand_uniform(double low, double high, UINT32_t* random_state) nogil:
    """
    Generate a random double in [low; high).
    """
    return ((high - low) * <double> our_rand_r(random_state) / <double> RAND_R_MAX) + low

# TODO: replace with rand_r replacement
cdef inline double get_random(int random_state) nogil:
    """
    Generate a random number between 0 and 1 sampled uniformly.
    """
    srand(time(NULL))  # do once?
    return rand() / RAND_MAX

# =============================================================================
# Stack data structure
# =============================================================================

cdef class Stack:
    """
    A LIFO data structure.

    Attributes
    ----------
    capacity : int
        The elements the stack can hold; if more added then ``self.stack_`` needs to be resized.
    top : int
        The number of elements currently on the stack.
    stack : StackRecord pointer
        The stack of records (upward in the stack corresponds to the right).
    """

    def __cinit__(self, int capacity):
        self.capacity = capacity
        self.top = 0
        self.stack_ = <StackRecord*> malloc(capacity * sizeof(StackRecord))

    def __dealloc__(self):
        free(self.stack_)

    cdef bint is_empty(self) nogil:
        return self.top <= 0

    cdef int push(self, int depth, int parent, bint is_left, int* samples,
                  int n_samples, int* features, int n_features) nogil except -1:
        """
        Push a new element onto the stack.
        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef int top = self.top
        cdef StackRecord* stack = NULL

        # Resize if capacity not sufficient
        if top >= self.capacity:
            self.capacity *= 2
            self.stack_ = <StackRecord *>realloc(self.stack_, self.capacity * sizeof(StackRecord))

            # # Since safe_realloc can raise MemoryError, use `except -1`
            # safe_realloc(&self.stack_, self.capacity)

        stack = self.stack_
        stack[top].depth = depth
        stack[top].parent = parent
        stack[top].is_left = is_left
        stack[top].samples = samples
        stack[top].n_samples = n_samples
        stack[top].features = features
        stack[top].n_features = n_features

        # Increment stack pointer
        self.top = top + 1
        return 0

    cdef int pop(self, StackRecord* res) nogil:
        """
        Remove the top element from the stack and copy to ``res``.
        Returns 0 if pop was successful (and ``res`` is set); -1 otherwise.
        """
        cdef int top = self.top
        cdef StackRecord* stack = self.stack_

        if top <= 0:
            return -1

        res[0] = stack[top - 1]
        self.top = top - 1

        return 0
