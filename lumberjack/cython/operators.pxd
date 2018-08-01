# -*- coding: utf-8 -*-

cimport numpy as np
from libcpp cimport bool
from libcpp.string cimport string
from .includes cimport DataPtr, DType

cdef extern from "./../rust/liblumberjack.h":

    DataPtr arange(int start, int stop, DType dtype)
    double  sum(DataPtr ptr)
    DataPtr cumsum(DataPtr ptr)
    double  mean(DataPtr ptr)
    DataPtr multiply_by_scalar(DataPtr ptr, double scalar, bool inplace)
    DataPtr add_by_scalar(DataPtr ptr, double scalar, bool inplace)
    DataPtr astype(DataPtr ptr, DType dtype)
    void set_item(DataPtr ptr, np.uint32_t idx, double value)
