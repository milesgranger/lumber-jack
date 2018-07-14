# -*- coding: utf-8 -*-
# distutils: language = c++

import logging

import numpy as np
cimport numpy as np

from cython cimport view
from .includes cimport free_series, arange, DataPtr, DType


logger = logging.getLogger(__name__)

np.import_array()


cdef LumberJackSeries create_lj_series_from_series_ptr(DataPtr ptr):
    series = LumberJackSeries()
    series.data_ptr = ptr.float64.data_ptr
    series.len = ptr.float64.len
    series.lj_series_ptr = ptr
    series.array_view = <double[:4]> ptr.float64.data_ptr
    return series

cdef class LumberJackSeries:

    cdef DataPtr lj_series_ptr
    cdef double* data_ptr
    cdef readonly int len
    cdef readonly view.array array_view


    @staticmethod
    def arange(int start, int stop):
        """
        This is ~2x faster than numpy's arange (tested 100000 times with range 0-100000)
        """
        cdef DataPtr ptr = arange(start, stop, DType.Float64)
        series = create_lj_series_from_series_ptr(ptr)
        return series


    def to_cython_array_view(self):
        """
        Provide a cython array view to the data
        """
        cdef view.array array_view = <double[:self.len]> self.data_ptr
        return array_view

    def to_numpy(self):
        """
        Convert this to numpy array
        """
        cdef np.ndarray array = np.asarray(<double[:self.len]> self.data_ptr)
        return array

    @staticmethod
    cdef from_lumberjack_ptr(DataPtr series_ptr):
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
        return 'LumberJackSeries(length: {})'.format(self.len)


    def __dealloc__(self):
        #logger.debug('Deallocating rust series!')
        if self.data_ptr != NULL:
            free_series(self.lj_series_ptr)


