# -*- coding: utf-8 -*-

import unittest

import lumberjack as lj
import pandas as pd


class LumberJackSeriesTestCase(unittest.TestCase):

    def setUp(self):
        self.lj_series = lj.Series(data=range(10))
        self.pd_series = pd.Series(data=range(10))
        pass

    def test_basic_construction(self):
        """Sanity check"""
        lj_series_sum = self.lj_series.sum()
        pd_series_sum = self.pd_series.sum()
        self.assertEqual(
            lj_series_sum, pd_series_sum,
            msg='Expected sum of pandas ({}) and lumberjack ({}) series to match.'.format(pd_series_sum, lj_series_sum)
        )

    def test_map_functionality(self):
        """
        Test that the Rust implementation on lumberjack.Series matches pandas.Series
        """
        lj_series_double = self.lj_series.map(lambda val: val * 2).sum()
        pd_series_double = self.pd_series.map(lambda val: val * 2).sum()
        self.assertEqual(
            pd_series_double, lj_series_double,
            msg='Expected sum of pandas ({}) and lumberjack ({}) series to match.'.format(pd_series_double, lj_series_double)
        )
