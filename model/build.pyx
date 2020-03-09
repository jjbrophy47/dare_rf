
# import numpy as np
# cimport numpy as np
# DTYPE = np.int
# ctypedef np.int_t DTYPE_t

import time
import numpy
cimport numpy
cimport cython

ctypedef numpy.int_t DTYPE_t


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire funct
def do_calc(numpy.ndarray[DTYPE_t, ndim=1] arr):
    cdef int maxval
    cdef unsigned long long int total
    cdef int k
    cdef double t1, t2, t
    cdef int arr_shape = arr.shape[0]

    t1=time.time()

#    for k in arr:
#        total = total + k

    for k in range(arr_shape):
        total = total + arr[k]
    print "Total =", total

    t2=time.time()
    t = t2-t1
    print("%.20f" % t)


def build(unsigned long long int maxval):

    cdef numpy.ndarray[DTYPE_t, ndim=1] arr = numpy.arange(maxval, dtype=numpy.int)
    cdef unsigned long long int total = 0
    cdef int k
    cdef double t1, t2, t

    cdef int n_samples = arr.shape[0]

    t1 = time.time()
    for k in range(n_samples):
        total += arr[k]
    t2 = time.time()
    t = t2 - t1
    print('time: %.20f' % t)
    print('total: {}'.format(total))


def build_py(maxval):
    import time
    import numpy as np
    arr = np.arange(maxval)
    total = 0
    t1 = time.time()
    for k in arr:
        total += k
    t2 = time.time()
    print('total: {}'.format(total))
    print('time: {:.3f}s'.format(t2 - t1))
