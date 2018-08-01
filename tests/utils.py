# -*- coding: utf-8 -*-

import logging
import timeit
import numpy as np

logger = logging.getLogger(__name__)


def run_series_method_tests(stmt: str,
                            alternate_numpy_stmt: str=None,
                            alternate_pandas_stmt: str=None,
                            skip_numpy: bool=False,
                            n_iter: int=2500):
    """
    Run the timing tests for LumberJack, Numpy and Pandas
    calling the same method on an arange of the series/array with an optional alternative
    to numpy (ie. np.abs(array), array.abs() doesn't exist)
    """
    # Test speed
    lj_time = timeit.timeit(stmt=stmt,
                            number=n_iter,
                            setup='import lumberjack as lj; series = lj.Series.arange(0, 50000)')
    pd_time = timeit.timeit(stmt=stmt if alternate_pandas_stmt is None else alternate_pandas_stmt,
                            number=n_iter,
                            setup='import numpy as np; import pandas as pd; series = pd.Series(np.arange(0, 50000))')

    if skip_numpy:
        np_time = np.NaN
    else:
        np_time = timeit.timeit(stmt=stmt if alternate_numpy_stmt is None else alternate_numpy_stmt,
                                number=n_iter,
                                setup='import numpy as np; series = np.arange(0, 50000)')

    logger.debug('"{}" speed: Avg LumberJack: {:.4f}s -- Pandas: {:.4f} -- Numpy: {:.4f}'
                 .format(stmt, lj_time, pd_time, np_time))
    return lj_time, pd_time, np_time
