cimport numpy as np

cdef extern from "./../rust/liblumberjack.h":

    # Allow rust to remove a vector that it created and passed to Python
    void free_vector(double* ptr, int length);

    cdef struct LumberJackSeriesPtr:
        double* data_ptr
        int len

    cdef cppclass LumberJackData:
        pass

    cdef enum DType:
        Float64 "DType::Float64"
        Int32 "DType::Float64"

    LumberJackSeriesPtr arange(int start, int stop, DType dtype);
    LumberJackSeriesPtr from_numpy_ptr(double * ptr, int length)



