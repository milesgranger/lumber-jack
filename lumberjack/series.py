# -*- coding: utf-8 -*-

import logging
import cloudpickle
import pandas as pd
from lumberjack.cython.series import LumberJackSeries

logger = logging.getLogger(__name__)


class Series(LumberJackSeries):
    pass
