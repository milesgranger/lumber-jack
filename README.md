# lumber-jack

![Build Status](https://travis-ci.org/milesgranger/lumber-jack.svg?branch=master)

---

This package strives to provide fast, safe and concurrent pre-processing tools for 
Python. The core computations are written in Rust and wrapped with intuitive Python 
interfaces.  

---

### Install

`cargo` & `rustup` should be installed along with nightly version of Rust  

**NOTE** Only Python 3.5 is being tested against on Unix platforms



From the terminal run:
```commandline
# Python 3.5 on Unix only at present
pip install --upgrade lumber-jack

# bleeding: (You need Rust on your system & Python >= 3.5)
rustup override set nightly
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