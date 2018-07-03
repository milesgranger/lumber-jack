# -*- coding: utf-8 -*-


import unittest
import logging
import numpy as np

logger = logging.getLogger(__name__)


class RustSeriesTestCase(unittest.TestCase):

    def test_sum_two_in_rust(self):
        """
        Check summing two values via Rust
        """
        from lumberjack.cython.series import sum_two
        result = sum_two(2, 4)
        logger.debug('Result: {}'.format(result))

    def test_double_numpy_array_in_rust(self):
        """
        Check passing a numpy array to rust and doubling it via pointers through C
        """
        from lumberjack.cython.series import double_array_in_rust
        array = np.ones(10, dtype=np.double)
        logger.debug('Array before passing to Rust: {}'.format(array))
        result = double_array_in_rust(array)
        logger.debug('Result of double array from rust: {}'.format(result))
        logger.debug('Here is the array in Python: {}'.format(array))
        logger.debug('Original Array Id: {}, Result Array Id: {}'.format(id(array), id(result)))

    def test_create_array_from_rust(self):
        """
        Check creating an array from inside Rust and passing it to Python
        """
        from lumberjack.cython.series import _create_array
        vec = _create_array()
        logger.debug('Vector type: {}, and it looks like: {}, sum is: {}'.format(type(vec), vec, vec.sum()))
        self.assertEqual(vec.sum(), 10)


    def test_free_array(self):
        """
        Test that when given a pointer to an lumberjack array, manually calling rust based 'free' will delete
        that array
        """
        from lumberjack.cython.series import LumberJackSeries, get_lumberjack_vector
        array = get_lumberjack_vector()

        logger.debug('Array before dropping: {}'.format(array))
        logger.debug('Array afer dropping: {}'.format(array))
        del array

