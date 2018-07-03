# -*- coding: utf-8 -*-


import unittest
import logging
import numpy as np

logger = logging.getLogger(__name__)


class RustSeriesTestCase(unittest.TestCase):

    def test_create_from_pandas(self):
        """
        Create Series from pandas series
        """
        from lumberjack.cython.series import sum_two
        result = sum_two(2, 4)
        logger.debug('Result: {}'.format(result))

    def test_pass_numpy_array(self):
        """
        Check passing a numpy array
        """
        from lumberjack.cython.series import double_array_in_rust
        array = np.ones(10, dtype=np.double)
        logger.debug('Array before passing to Rust: {}'.format(array))
        result = double_array_in_rust(array)
        logger.debug('Result of double array from rust: {}'.format(result))
        logger.debug('Here is the array in Python: {}'.format(array))