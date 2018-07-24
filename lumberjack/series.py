# -*- coding: utf-8 -*-

import logging
import cloudpickle
import pandas as pd
from lumberjack.cython.series import LumberJackSeries

logger = logging.getLogger(__name__)


class Series(LumberJackSeries):

    pass

    #def map(self, func: callable):
    #    func = cloudpickle.dumps(func)
    #    return super(self, LumberJackSeries)._map(func)

