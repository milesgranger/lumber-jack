# -*- coding: utf-8 -*-

import unittest
import logging
import timeit
import numpy as np
import pandas as pd

import lumberjack as lj


logger = logging.getLogger(__name__)


class RustSeriesDundersTestCase(unittest.TestCase):
    """
    Test double under impls.
    """
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
