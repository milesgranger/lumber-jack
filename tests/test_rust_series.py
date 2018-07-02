# -*- coding: utf-8 -*-


import unittest
import logging

logger = logging.getLogger(__name__)


class RustSeriesTestCase(unittest.TestCase):

    def test_create_from_pandas(self):
        """
        Create Series from pandas series
        """
        from lumberjack.cython.series import sum_two
        result = sum_two(2, 4)
        logger.debug('Result: {}'.format(result))
