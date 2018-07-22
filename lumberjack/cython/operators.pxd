# -*- coding: utf-8 -*-

from libcpp cimport bool
from .includes cimport DataPtr, DType

cdef extern from "./../rust/liblumberjack.h":

    DataPtr arange(int start, int stop, DType dtype)
    DataPtr sum(DataPtr ptr)
    DataPtr cumsum(DataPtr ptr)
    double  mean(DataPtr ptr)
    DataPtr multiply_by_scalar(DataPtr ptr, double scalar)
    void*   imultiply_by_scalar(DataPtr* ptr, double scalar)
    DataPtr add_by_scalar(DataPtr ptr, double scalar)
    void*   iadd_by_scalar(DataPtr* ptr, double scalar)