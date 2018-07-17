cimport numpy as np

cdef extern from "./../rust/liblumberjack.h":

    cdef enum DType:
        Float64 "DType::Float64"
        Int32 "DType::Int32"

    cdef enum Tag "DataPtr::Tag":
        Tag_Float64 "DataPtr::Tag::Float64"
        Tag_Int32   "DataPtr::Tag::Int32"


    cdef struct Float64_S "DataPtr::Float64_Body":
        double* data_ptr
        int len


    cdef struct Int32_S "DataPtr::Int32_Body":
        np.int32_t* data_ptr
        int len


    cdef union dataptr_union "DataPtr":
        Float64_S float64
        Int32_S int32

    cdef struct DataPtr:
        Tag tag
        Float64_S float64
        Int32_S int32

    # Allow rust to remove a vector that it created and passed to Python
    void free_data(DataPtr ptr);
