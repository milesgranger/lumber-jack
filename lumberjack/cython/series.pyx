# -*- coding: utf-8 -*-
# distutils: language = c++

import logging
import numpy as np
import pandas as pd

cimport numpy as np

from libcpp cimport bool
from cython cimport view
from .includes cimport free_data, DataPtr, DType, Tag
from .operators cimport arange, sum as _sum, cumsum, mean, multiply_by_scalar, add_by_scalar

logger = logging.getLogger(__name__)

np.import_array()


cdef LumberJackSeries create_lj_series_from_data_ptr(DataPtr ptr):
    """ 
    Factory for creating LumberJackSeries from DataPtr
    **cannot be used as classmethod from within LumberJackSeries**
    """
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
    """
    Holds generic access to various data types from a DataPtr
    """
    # Possible array pointers for different dtypes
    cdef double* vec_ptr_float64
    cdef np.int32_t* vec_ptr_int32

    # Static attrs across all dtypes of a DataPtr object.
    cdef readonly view.array array_view
    cdef readonly int len
    cdef DataPtr data_ptr

    def __dealloc__(self):
        if self.vec_ptr_float64 != NULL or \
                self.vec_ptr_int32 != NULL:
            free_data(self.data_ptr)

cdef class LumberJackSeries:
    """
    LumberJackSeries

    Some implementations of Numpy / Pandas functionality with bindings to Rust.
    """
    cdef _DataPtr _data_ptr

    def _scalar_arithmetic_factory(self, double scalar, str op, bool inplace):
        cdef DataPtr ptr
        if op == 'mul':
            ptr = multiply_by_scalar(self._data_ptr.data_ptr, scalar, inplace)
        elif op == 'add':
            ptr = add_by_scalar(self._data_ptr.data_ptr, scalar, inplace)
        else:
            raise ValueError('Unknown operation: {}'.format(op))
        return create_lj_series_from_data_ptr(ptr)

    def __mul__(self, other):
        return self._scalar_arithmetic_factory(float(other), 'mul', False)

    def __imul__(self, other):
        self = self._scalar_arithmetic_factory(float(other), 'mul', True)
        return self

    def __add__(self, other):
        return self._scalar_arithmetic_factory(float(other), 'add', False)

    def __iadd__(self, other):
        self = self._scalar_arithmetic_factory(float(other), 'add', True)
        return self

    @staticmethod
    def arange(int start, int stop):
        """
        This is ~2x faster than numpy's arange (tested 100000 times with range 0-100000)
        """
        cdef DataPtr ptr = arange(start, stop, DType.Int32)
        return create_lj_series_from_data_ptr(ptr)

    def mean(self):
        cdef double avg
        avg = mean(self._data_ptr.data_ptr)
        return avg

    def sum(self):
        cdef DataPtr ptr = _sum(self._data_ptr.data_ptr)
        cdef LumberJackSeries series = create_lj_series_from_data_ptr(ptr)
        return series._data_ptr.array_view[0]

    def cumsum(self):
        cdef DataPtr ptr = cumsum(self._data_ptr.data_ptr)
        return create_lj_series_from_data_ptr(ptr)

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

    def __len__(self):
        return self._data_ptr.len

    def __iter__(self):
        return (self._data_ptr.array_view[i] for i in range(self._data_ptr.len))

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

    def __getattr__(self, item):
        def method(*args, **kwargs):
            series = pd.Series(self.to_numpy())  # TODO: pass more features to constructor as we add them.
            return getattr(series, item)(*args, **kwargs)
        return method

    def __repr__(self):
        return 'LumberJackSeries(length: {})'.format(self._data_ptr.len)


