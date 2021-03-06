# -*- coding: utf-8 -*-

import unittest
import logging
import timeit
import time
import pickle
import cloudpickle
import numpy as np
import pandas as pd
import lumberjack as lj

from tests.utils import run_series_method_tests


logger = logging.getLogger(__name__)


class RustSeriesTestCase(unittest.TestCase):

    def test_from_numpy(self):
        array = np.arange(0, 10, dtype=float)
        series = lj.Series.from_numpy(array)
        self.assertEqual(array.sum(), series.sum())

    def test_astype(self):
        """Test converting series from one type to another"""
        int_series = lj.Series.arange(0, 50)
        float_series = int_series.astype(float)
        self.assertEqual(int_series.to_numpy().dtype, np.int32)
        self.assertEqual(float_series.to_numpy().dtype, np.float64)

    def test_series_map(self):
        lj_series = lj.Series.arange(0, 10, float)
        variable = 2.0
        self.assertNotEqual(lj_series.sum(), 0.0)
        #result = lj_series.map(lambda v: v * 0.0, out_dtype=float)
        #self.assertEqual(result.sum(), 0.0)
        #logger.debug('Result values: {}'.format(result.to_numpy().astype(int)))
        #result1 = lj_series.map(lambda v: 2.0, out_dtype=float)
        #result2 = lj_series.map(lambda v: 30, out_dtype=float)
        #logger.debug('Result from .map() -> {}'.format(result))
        lj_time, pd_time, _ = run_series_method_tests(stmt="series.map(lambda v: v)", skip_numpy=True, n_iter=20)

    def test_picklable(self):
        """
        Test ability to be pickled.
        """
        series = lj.Series.arange(0, 10)
        pkl = pickle.dumps(series)
        series_copy = pickle.loads(pkl)
        self.assertEqual(series.sum(), series_copy.sum())
        self.assertFalse(series_copy.is_owner)

        # Assert deleting the copy doesn't affect the original
        orig_sum = series.sum()
        del series_copy
        self.assertEqual(series.sum(), orig_sum)

        # Do another pickle.. jbc
        pkl = pickle.dumps(series)
        series_copy = pickle.loads(pkl)
        self.assertEqual(series.sum(), series_copy.sum())
        self.assertFalse(series_copy.is_owner)

    def test_mean(self):
        """
        Test average/mean of series
        """
        lj_series = lj.Series.arange(0, 10000)
        pd_series = pd.Series(np.arange(0, 10000))
        avg = lj_series.mean()
        logger.debug('Mean of arange(0, 10000) -> {:.4f}'.format(avg))
        self.assertEqual(lj_series.mean(), pd_series.mean())

        # Speed test
        lj_time, pd_time, np_time = run_series_method_tests('series.mean()')
        self.assertLessEqual(lj_time, pd_time)

    def test_cumsum(self):
        """
        Test cumulative sum of series
        """
        series = lj.Series.arange(0, 4)
        _pd_series = pd.Series(np.arange(0, 4))
        cumsum = series.cumsum()
        logger.debug('Got cumsum of {}'.format(cumsum))

        # Ensure they sum to the same
        lj_cumsum_sum = series.cumsum().sum()
        pd_cumsum_sum = _pd_series.cumsum().sum()
        self.assertEqual(lj_cumsum_sum, pd_cumsum_sum,
                         msg='LumberJack and Pandas .cumsum().sum() does not match! -- LumberJack: {}, Pandas: {}'
                             .format(lj_cumsum_sum, pd_cumsum_sum)
                         )
        # Speed test
        lj_time, pd_time, np_time = run_series_method_tests('series.cumsum()')
        self.assertLessEqual(lj_time, pd_time)


    def test_sum(self):
        """
        Test the ability to sum a series
        """
        series = lj.Series.arange(0, 4)
        total = series.sum()
        logger.debug('Sum of arange(0, 4) is: {}'.format(total))
        self.assertEqual(total, 6)

        # Speed test
        lj_time, pd_time, np_time = run_series_method_tests('series.sum()')
        self.assertLessEqual(lj_time, np_time, msg='Expected LumberJack .sum() to be faster but it was not!')

    def test_arange(self):
        """
        Check creating an array from inside Rust and passing it to Python
        """
        series = lj.Series.arange(0, 4)
        vec = series.to_numpy()
        logger.debug('Vector type: {}, and it looks like: {}, sum is: {}'.format(type(vec), vec, vec.sum()))
        self.assertEqual(vec.sum(), 6)

        # If we re-implement numpy function, they should be faster
        lj_time = timeit.timeit('LumberJackSeries.arange(0, 10000)',
                                number=10000,
                                setup='from lumberjack.cython.series import LumberJackSeries')
        np_time = timeit.timeit('np.arange(0, 10000)',
                                number=10000,
                                setup='import numpy as np')
        logger.debug('.arange(0, 10000) speed: Avg LumberJack: {:4f}s -- Avg numpy: {:.4f}s'.format(lj_time, np_time))
        self.assertLess(lj_time, np_time,
                        'Expected LumberJack ({:.4f}) to be faster than numpy ({:.4f}), but it was not!'
                        .format(lj_time, np_time))

    def test_describe(self):
        """
        Test that a call to a method not implemented in LumberJack will forward that call to Pandas
        in this case '.describe()'

        Additionally, ensure a call to a method with neither has, raises the same error as it would otherwise.
        """
        series = lj.Series.arange(0, 10)
        descript = series.describe()
        self.assertIsInstance(
            obj=descript,
            cls=pd.Series,
            msg='Expected call to ".describe()" to produce a pandas Series, it produced a {} instead!'.format(descript)
        )
        with self.assertRaises(AttributeError):
            series.THIS_METHOD_DOES_NOT_EXIST_IN_EITHER_PANDAS_OR_LUMBERJACK()
    '''
    def test_from_numpy(self):
        """
        Test creating a series from numpy array
        """
        from lumberjack.cython.series import LumberJackSeries

        array = np.ones(shape=(10,), dtype=float)
        series = LumberJackSeries.from_numpy(array)
        logger.debug('Made series from numpy: {}'.format(series))
    '''
