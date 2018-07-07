# -*- coding: utf-8 -*-


import unittest
import logging
import numpy as np
import pandas as pd
import timeit

logger = logging.getLogger(__name__)


class RustSeriesTestCase(unittest.TestCase):

    def test_boxed_int(self):
        from lumberjack.cython.series import get_boxed_int
        value = get_boxed_int()
        logger.debug('Boxed value: {}'.format(value))

    def test_from_arange(self):
        """
        Check creating an array from inside Rust and passing it to Python
        """
        from lumberjack.cython.series import LumberJackSeries
        series = LumberJackSeries.arange(0, 4)
        vec = series.to_numpy()
        logger.debug('Vector type: {}, and it looks like: {}, sum is: {}'.format(type(vec), vec, vec.sum()))
        self.assertEqual(vec.sum(), 6)

        # If we re-implement numpy function, they should be faster
        lj_time = timeit.timeit('LumberJackSeries.arange(0, 10000)', number=10000, setup='from lumberjack.cython.series import LumberJackSeries')
        np_time = timeit.timeit('np.arange(0, 10000)', number=10000, setup='import numpy as np')
        logger.debug('Avg time for LumberJack arange: {:4f}'.format(lj_time))
        logger.debug('Avg time for numpy arange: {:4f}'.format(np_time))
        self.assertLess(lj_time, np_time,
                        'Expected LumberJack ({:.4f}) to be faster than numpy ({:.4f}), but it was not!'
                        .format(lj_time, np_time))

    def test_from_numpy(self):
        """
        Test creating a series from numpy array
        """
        from lumberjack.cython.series import LumberJackSeries

        array = np.ones(shape=(10,), dtype=float)
        series = LumberJackSeries.from_numpy(array)
        logger.debug('Made series from numpy: {}'.format(series))



