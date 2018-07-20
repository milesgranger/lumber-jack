# -*- coding: utf-8 -*-

import unittest
import logging

import lumberjack as lj
from tests.utils import run_series_method_tests

logger = logging.getLogger(__name__)


class RustSeriesDundersTestCase(unittest.TestCase):
    """
    Test double under impls.
    """

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
