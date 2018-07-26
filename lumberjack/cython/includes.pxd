cimport numpy as np

cdef extern from "./../rust/liblumberjack.h":


    # START - Define DataElement Struct....
    cdef enum TagDataElement "DataElement::Tag":
        TagDataElement_Float64 "DataElement::Tag::Float64"
        TagDataElement_Int32 "DataElement::Tag::Int32"

    cdef struct Float64_DataElement "DataElement::Float64_Body":
        double item "DataElement::Float64_Body::_0"

    cdef struct Int32_DataElement "DataElement::Int32_Body":
        np.int32_t item "DataElement::Float64_Body::_0"

    cdef union data_element_union "DataElement":
        Float64_DataElement float64
        Int32_DataElement int32

    cdef struct DataElement:
        TagDataElement tag
        Float64_DataElement float64
        Int32_DataElement int32
    # END - Define DataElement struct



    # START - Define DType enum
    cdef enum DType:
        Float64 "DType::Float64"
        Int32 "DType::Int32"
    # END - Define DType enum



    # START - Define DataPtr struct...
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
    # END - Define DataPtr struct

    # Allow rust to remove a vector that it created and passed to Python
    void free_data(DataPtr ptr);
    void verify(DataPtr ptr);
