# -*- coding: utf-8 -*-

import logging
import timeit

logger = logging.getLogger(__name__)


def run_series_method_tests(stmt: str, alternate_numpy_stmt: str=None):
    """
    Run the timing tests for LumberJack, Numpy and Pandas
    calling the same method on an arange of the series/array with an optional alternative
    to numpy (ie. np.abs(array), array.abs() doesn't exist)
    """
    # Test speed
    lj_time = timeit.timeit(stmt=stmt,
                            number=10000,
                            setup='import lumberjack as lj; series = lj.Series.arange(0, 10000)')
    pd_time = timeit.timeit(stmt=stmt,
                            number=10000,
                            setup='import numpy as np; import pandas as pd; series = pd.Series(np.arange(0, 10000))')
    np_time = timeit.timeit(stmt=stmt if alternate_numpy_stmt is None else alternate_numpy_stmt,
                            number=10000,
                            setup='import numpy as np; series = np.arange(0, 10000)')

    logger.debug('"{}" speed: Avg LumberJack: {:.4f}s -- Pandas: {:.4f} -- Numpy: {:.4f}'
                 .format(stmt, lj_time, pd_time, np_time))
    return lj_time, pd_time, np_time
