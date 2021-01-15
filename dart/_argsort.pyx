cimport cython
from libc.stdlib cimport malloc
from libc.stdlib cimport free
 
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil

cdef struct IndexedElement:
    int    index
    double value

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int _compare(const_void *a, const_void *b):
    cdef double v = (<IndexedElement*> a).value-(<IndexedElement*> b).value
    if v < 0:
        return -1
    if v >= 0:
        return 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void argsort(double* data, int* output, int n_samples) nogil:
    """
    Returns index array of sorted elements.
    """
    cdef int i
    
    # index tracking array
    cdef IndexedElement *order_struct = <IndexedElement *> malloc(n_samples * sizeof(IndexedElement))
    
    # copy data into index tracking array
    for i in range(n_samples):
        order_struct[i].index = i
        order_struct[i].value = data[i]
        
    # sort index tracking array.
    qsort(<void *> order_struct, n_samples, sizeof(IndexedElement), _compare)
    
    # copy indices from index tracking array to order array
    for i in range(n_samples):
        output[i] = order_struct[i].index
        
    # free index tracking array
    free(order_struct)
    

        

