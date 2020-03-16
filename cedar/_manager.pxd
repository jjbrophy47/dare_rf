import numpy as np
cimport numpy as np

cdef class _DataManager:
    """
    Manages the database.
    """

    # Internal structure
    cdef int   n_samples       # Number of samples
    cdef int   n_features      # Number of features
    cdef int   n_vacant        # Number of empty indices in the database
    cdef int** X               # Sample data
    cdef int*  y               # Label data
    cdef int*  f               # Features
    cdef int*  vacant          # Empty indices in the database

    # C API
    cdef int check_sample_validity(self, int *samples, int n_samples) nogil
    cdef void get_data(self, int*** X_ptr, int** y_ptr) nogil
    cdef void get_features(self, int** f_ptr) nogil
    # cdef int get_all_data(self, int*** X_ptr, int** y_ptr, int**f_ptr,
    #                       int* n_samples, int* n_features) nogil
    # cdef int get_data_subset(self, int* samples, int n_samples,
    #                          int ***X_sub_ptr, int **y_sub_ptr) nogil
    cdef int remove_data(self, int* samples, int n_samples) nogil
