# -*- coding: utf-8 -*-

import numpy as np
from collections import Iterable
from lumberjack.rust import alterations  # rust compiled module


def split_n_one_hot_encode(raw_texts: Iterable, sep: str, cutoff: int):
    """
    Given array of raw texts where each instance is separated by <something> return one-hot encoded array where
    each instance of the one-hot array indicates if the word was present in the instance of raw_text

    :param raw_texts:
    :param sep:
    :param cutoff:
    :return:
    """

    if not isinstance(raw_texts, Iterable):
        raise ValueError('raw_texts should be an iterable with values string values')

    if not isinstance(sep, str):
        raise ValueError('sep should be a str instance')

    if not isinstance(cutoff, int):
        raise ValueError('cutoff should be an integer')

    words, array = alterations.split_n_one_hot_encode(raw_texts, sep, cutoff)
    return words, np.array(array, dtype=np.uint8)
