# -*- coding: utf-8 -*-

import unittest
import logging

import lumberjack as lj
from tests.utils import run_series_method_tests

logger = logging.getLogger(__name__)


class RustSeriesDundersTestCase(unittest.TestCase):
    """
    Test dunder impls.
    """

    def test_indexing_getitem(self):
        """Depends on __getitem__"""
        series = lj.Series.arange(0, 5)
        self.assertEqual(series[0], 0)
        self.assertEqual(series[-1], 4)

    def test_indexing_setitem(self):
        """Depends on __setitem__"""
        series = lj.Series.arange(0, 5)

        # Test basic capability
        self.assertEqual(series[0], 0)
        series[0] = 1
        self.assertEqual(series[0], 1)

        # Test adding float to Int32, should convert any decimal to it's base integer value as numpy does.
        series[0] = 10.9
        self.assertEqual(series[0], 10)

        # Test adding string to int32 series raises a TypeError
        with self.assertRaises(ValueError):
            series[0] = 'String does not belong in this int32 series!'

    def test_sum(self):
        """Depends on __iter__"""
        lj_series = lj.Series.arange(0, 5)
        self.assertEqual(lj_series.sum(), sum(lj_series))

    def test_add_by_scalar(self):
        lj_series = lj.Series.arange(0, 5)
        result = lj_series + 1
        self.assertEqual(result.sum(), 15)

        # Speed Test
        run_series_method_tests('series + 1')

    def test_add_by_scalar_inplace(self):
        lj_series = lj.Series.arange(0, 5)
        lj_series += 1
        self.assertEqual(lj_series.sum(), 15)

        # Speed Test
        run_series_method_tests('series += 1')

    def test_multiply_by_scalar(self):
        lj_series = lj.Series.arange(0, 5)
        result = lj_series * 2
        self.assertEqual(result.sum(), 20)

        # Speed Test
        run_series_method_tests('series * 2')

    def test_multiply_by_scalar_inplace(self):
        lj_series = lj.Series.arange(0, 5)
        lj_series *= 2
        self.assertEqual(lj_series.sum(), 20)

        # Speed Test
        run_series_method_tests('series *= 2')

    def test_len(self):
        """
        Test __len__, length check
        """
        series = lj.Series.arange(0, 5)
        self.assertEqual(len(series), 5)

    def test_iter(self):
        """
        Test appropriate iteration capability
        """
        series = lj.Series.arange(0, 5)
        _list = list(range(5))

        for v1, v2 in zip(series, _list):
            self.assertEqual(v1, v2)
