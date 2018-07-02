# -*- coding: utf-8 -*-


import unittest


class RustSeriesTestCase(unittest.TestCase):

    def test_create_from_pandas(self):
        """
        Create Series from pandas series
        """
        from lumberjack.cython.series import sum_two
