import numpy as np
cimport numpy as np

cdef class _DataManager:
    """
    Manages the database.
    """

    # Internal structure
    cdef int   n_samples       # Number of samples
    cdef int   n_vacant        # Number of empty indices in the database
    cdef int** X               # Sample data
    cdef int*  y               # Label data
    cdef int*  vacant          # Empty indices in the database

    # Python API
    cpdef int remove_data(self, int[:] samples)

    # C API
    cdef int check_sample_validity(self, int *samples, int n_samples) nogil
    cdef void get_data(self, int*** X_ptr, int** y_ptr) nogil
