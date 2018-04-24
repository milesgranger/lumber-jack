# -*- coding: utf-8 -*-


import pandas as pd


class DataFrame(pd.DataFrame):

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
