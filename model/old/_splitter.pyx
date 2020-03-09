from libc.stdlib cimport free

import numpy as np
cimport numpy as np
np.import_array()

from utils cimport RAND_R_MAX
from utils cimport safe_realloc

cdef class Splitter:
    """
    Abstract splitter class.
    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.
        max_features : SIZE_t
            The maximal number of randomly selected features which can be
            considered for a split.
        min_samples_leaf : SIZE_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.
        random_state : object
            The user inputted random state to be used for pseudo-randomness
        """
        self.criterion = criterion

        self.samples = NULL
        self.n_samples = 0
        self.features = NULL
        self.n_features = 0
        self.feature_values = NULL

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def __dealloc__(self):
        """Destructor."""
        free(self.samples)
        free(self.features)
        free(self.feature_values)

    cdef int init(self, object X, const DOUBLE_t[:, ::1] y) except -1:
        """
        Initialize the splitter.
        Take in the input data X and the target Y.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.
        y : ndarray, dtype=DOUBLE_t
            This is the vector of targets, or true labels, for the samples
        """
        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef SIZE_t n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)
        self.n_samples = X.shape[0]

        cdef SIZE_t i
        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t* features = safe_realloc(&self.features, n_features)

        for i in range(n_features):
            features[i] = i

        self.n_features = n_features

        safe_realloc(&self.feature_values, n_samples)

        self.y = y

        return 0

    cdef int node_reset(self, SIZE_t start, SIZE_t end) nogil except -1:
        """
        Reset splitter on node samples[start:end].
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : SIZE_t
            The index of the first sample to consider
        end : SIZE_t
            The index of the last sample to consider
        """
        self.start = start
        self.end = end
        self.criterion.init(self.y, self.samples, start, end)
        return 0

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """
        Find the best split on node samples[start:end].
        This is a placeholder method. The majority of computation will be done here.
        It should return -1 upon errors.
        """

        pass

    cdef void node_value(self, double* dest) nogil:
        """
        Copy the value of node samples[start:end] into dest.
        """
        self.criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        """
        Return the impurity of the current node.
        """

cdef class BaseDenseSplitter(Splitter):
    cdef const DTYPE_t[:, :] X

    cdef np.ndarray X_idx_sorted
    cdef INT32_t* X_idx_sorted_ptr
    cdef SIZE_t X_idx_sorted_stride
    cdef SIZE_t n_total_samples
    cdef SIZE_t* sample_mask

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, object random_state):

        self.X_idx_sorted_ptr = NULL
        self.X_idx_sorted_stride = 0
        self.sample_mask = NULL
        self.criterion.node_impurity()

    cdef int init(self, object X, const DOUBLE_t[:, ::1] y) except -1:
        """
        Initialize the splitter
        Returns -1 in case of failure to allocate memory (and raise MemoryError) or 0 otherwise.
        """

        # Call parent init
        Splitter.init(self, X, y)
        self.X = X
        return 0
