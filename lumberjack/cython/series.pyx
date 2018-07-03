# -*- coding: utf-8 -*-
# distutils: language = c++

from .includes cimport add_two_in_rust, double_array
import numpy as np


cpdef float sum_two(float a, float b):
    cdef float result
    result = add_two_in_rust(a, b)
    return result


cpdef double_array_in_rust(arr):

    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr) # Makes a contiguous copy of the numpy array.

    cdef double[::1] arr_view = arr
    double_array(&arr_view[0])
    return arr