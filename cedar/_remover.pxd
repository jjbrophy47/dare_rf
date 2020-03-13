import numpy as np
cimport numpy as np

from ._splitter cimport Meta

cdef struct RemovalSplitRecord:
    # Data to track sample split
    int* left_indices         # Samples in left branch of feature.
    int  left_count           # Number of samples in left branch.
    int* right_indices        # Samples in right branch of feature.
    int  right_count          # Number of samples in right branch.

cdef class _Remover:
    """
    Recursively removes data from a _Tree built using _TreeBuilder; retrains
    when the indistinguishability bounds epsilon has been violated.
    """

    # Inner structures
    cdef double epsilon                     # Indistinguishability parameter

    # Python API
    cpdef int remove(self, _Tree tree, _TreeBuilder tree_builder,
                     object X, np.darray y, np.ndarray samples)

    # C API
    cdef _check_input(self, object X, np.ndarray y)
    cdef int _node_remove(self, int[:, ::1] X, int[::1] y,
                         int* samples, int n_samples,
                         int min_samples_split, int min_samples_leaf, 
                         int chosen_feature, double parent_p,
                         RemovalSplitRecord *split, Meta* meta) nogil
    cdef int _collect_leaf_samples(self, _Tree tree, int node_id, int* samples) nogil
