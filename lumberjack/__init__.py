# -*- coding: utf-8 -*-

from .dataframe import DataFrame
from .series import Series

# TODO: Import other pd.* objects & functions which haven't been re-implemented dynamically.

MAJOR_VERSION = 0
MINOR_VERSION = 1
PATCH_VERSION = '1a'

__version__ = "{}.{}.{}".format(MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)
