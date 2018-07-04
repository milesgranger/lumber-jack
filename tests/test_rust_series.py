# -*- coding: utf-8 -*-


import unittest
import logging
import numpy as np

logger = logging.getLogger(__name__)


class RustSeriesTestCase(unittest.TestCase):

    def test_create_array_from_rust(self):
        """
        Check creating an array from inside Rust and passing it to Python
        """
        from lumberjack.cython.series import LumberJackSeries
        series = LumberJackSeries.linspace(0, 4)
        vec = series.to_numpy()
        logger.debug('Vector type: {}, and it looks like: {}, sum is: {}'.format(type(vec), vec, vec.sum()))
        self.assertEqual(vec.sum(), 10)

    def test_from_numpy(self):
        """
        Test that when given a pointer to an lumberjack array, manually calling rust based 'free' will delete
        that array
        """
        from lumberjack.cython.series import LumberJackSeries

        array = np.ones(shape=(10,), dtype=float)
        series = LumberJackSeries.from_numpy(array)
        logger.debug('Made series from numpy: {}'.format(series))



