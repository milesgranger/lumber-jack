cimport numpy as np

cdef extern from "./../rust/liblumberjack.h":

    cdef enum DType:
        Float64 "DType::Float64"
        Int32 "DType::Int32"

    cdef enum Tag "SeriesPtr::Tag":
        Tag_Float64 "SeriesPtr::Tag::Float64"
        Tag_Int32   "SeriesPtr::Tag::Int32"


    cdef struct Float64_S "SeriesPtr::Float64_Body":
        double* data_ptr
        int len


    cdef struct Int32_S "SeriesPtr::Int32_Body":
        np.int32_t* data_ptr
        int len


    cdef union dataptr_union "SeriesPtr":
        Float64_S float64
        Int32_S int32

    cdef struct SeriesPtr:
        Tag tag
        Float64_S float64
        Int32_S int32


    SeriesPtr arange(int start, int stop, DType dtype)

    # Allow rust to remove a vector that it created and passed to Python
    void free_series(SeriesPtr ptr);
