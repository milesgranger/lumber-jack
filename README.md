# lumber-jack

![Build Status](https://travis-ci.org/milesgranger/lumber-jack.svg?branch=master)

---

This package strives to provide fast, safe and concurrent pre-processing tools for 
Python. The core computations are written in Rust and wrapped with intuitive Python 
interfaces.  

---

### Install

`cargo` & `rustup` should be installed along with nightly version of Rust 2018-01-01 
or later set for the current directory.

From the terminal run:
```commandline
rustup override set nightly-2018-01-01
pip install git+https://github.com/milesgranger/lumber-jack.git


# Uninstall
pip uninstall lumber-jack
```

---

### Working Tools:

Split arrays of text which are separated by < something > into a one-hot encoded ndarray

```python
>>> from lumberjack import alterations
>>> raw_texts = ['hello, there', 'hi, there']
>>> alterations.split_n_one_hot_encode(raw_texts, sep=',', cutoff=0)
(['hello', 'there', 'hi'], [[1, 1, 0], [0, 1, 1]])

```

**Note**: This is also a package for attempting to re-implement a pandas-like tool, but this 
is _far_ in the future...in a land far, far away. :)