# -*- coding: utf-8 -*-

from .includes cimport DataPtr, DType

cdef extern from "./../rust/liblumberjack.h":

    DataPtr arange(int start, int stop, DType dtype)
    DataPtr sum(DataPtr ptr)
    DataPtr cumsum(DataPtr ptr)