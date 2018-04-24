# -*- coding: utf-8 -*-


import pandas as pd


class Series(pd.Series):

    def __init__(self, data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):
        super().__init__(data=data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath)
