# -*- coding: utf-8 -*-
# distutils: language = c++

import logging

import numpy as np
cimport numpy as np

from cython cimport view
from .includes cimport free_data, arange, DataPtr, DType, Tag


logger = logging.getLogger(__name__)

np.import_array()


cdef LumberJackSeries create_lj_series_from_data_ptr(DataPtr ptr):
    series = LumberJackSeries()
    cdef _DataPtr _data_ptr
    _data_ptr = _DataPtr()

    if ptr.tag == Tag.Tag_Float64:
        _data_ptr.vec_ptr_float64 = ptr.float64.data_ptr
        _data_ptr.array_view = <double[:ptr.float64.len]> ptr.float64.data_ptr
        _data_ptr.len = ptr.float64.len

    elif ptr.tag == Tag.Tag_Int32:
        _data_ptr.vec_ptr_int32 = ptr.int32.data_ptr
        _data_ptr.array_view = <np.int32_t[:ptr.int32.len]> ptr.int32.data_ptr
        _data_ptr.len = ptr.int32.len
        
    else:
        raise ValueError('Got unknown Dtype: {}'.format(ptr.tag))

    _data_ptr.data_ptr = ptr
    series._data_ptr = _data_ptr
    return series

cdef class _DataPtr:

    cdef double* vec_ptr_float64
    cdef np.int32_t* vec_ptr_int32

    cdef readonly view.array array_view
    cdef readonly int len
    cdef DataPtr data_ptr

    def __dealloc__(self):
        if self.vec_ptr_float64 != NULL or self.vec_ptr_int32 != NULL:
            free_data(self.data_ptr)

cdef class LumberJackSeries:

    cdef _DataPtr _data_ptr

    @staticmethod
    def arange(int start, int stop):
        """
        This is ~2x faster than numpy's arange (tested 100000 times with range 0-100000)
        """
        cdef DataPtr ptr = arange(start, stop, DType.Int32)
        return create_lj_series_from_data_ptr(ptr)

    def sum(self):
        return np.asarray(self._data_ptr.array_view).sum()


    def to_cython_array_view(self):
        """
        Provide a cython array view to the data
        """
        return self._data_ptr.array_view

    def to_numpy(self):
        """
        Convert this to numpy array
        """
        cdef np.ndarray array = np.asarray(self._data_ptr.array_view)
        return array

    @staticmethod
    cdef from_lumberjack_ptr(DataPtr data_ptr):
        """
        Create series from Cython/C struct of LumberJackSeriesPtr which holds meta
        data about the underlying Rust vector
        """
        return 'Done!'

    @staticmethod
    def from_numpy(np.ndarray array):
        """
        Create series from 1d numpy array
        """
        if array.ndim > 1:
            raise ValueError('Cannot make Series from an array with >1 shape, needs to be 1 dimensional. '
                             'The passed array had shape: {}'.format(array.ndim))

        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array) # Makes a contiguous copy of the numpy array.

        cdef double[::1] arr_view = array

        #series_ptr = from_numpy_ptr(&arr_view[0], array.shape[0])
        #series = create_lumberjack_series_from_ptr(series_ptr)
        #return series

    def __repr__(self):
        return 'LumberJackSeries(length: {})'.format(self._data_ptr.len)


