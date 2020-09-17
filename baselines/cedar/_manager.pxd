import numpy as np
cimport numpy as np

cdef class _DataManager:
    """
    Manages the database.
    """

    # Internal structure
    cdef int   n_samples       # Number of samples
    cdef int   n_features      # Number of features
    cdef int** X               # Sample data
    cdef int*  y               # Label data
    cdef int   n_vacant        # Number of empty indices in the database
    cdef int*  vacant          # Empty indices in the database
    cdef int*  add_indices     # Added indices in the database
    cdef int   n_add_indices   # Number of indices added to the database

    # Python API
    cpdef void remove_data(self, int[:] samples)
    cpdef void add_data(self, int[:, :] X_in, int[:] y_in)
    cpdef void clear_add_indices(self)

    # C API
    cdef int check_sample_validity(self, int *samples, int n_samples) nogil
    cdef void get_data(self, int*** X_ptr, int** y_ptr) nogil
    cdef int* get_add_indices(self) nogil
    cdef np.ndarray _get_int_ndarray(self, int *data, int n_elem)
