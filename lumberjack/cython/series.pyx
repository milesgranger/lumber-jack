# -*- coding: utf-8 -*-
# distutils: language = c++


from .includes cimport add_two_in_rust, double_array, create_array
import numpy as np
cimport numpy as np


np.import_array()


cpdef create_array_via_rust():
    vector = create_array()
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> vector.len

    array = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, vector.data)
    return array


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