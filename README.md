![](logo.png)

---

[![Build Status](https://travis-ci.org/milesgranger/lumber-jack.svg?branch=master)](https://travis-ci.org/milesgranger/lumber-jack)

---

First and foremost: This is a project I'm using to refine my craft utilizing Python, Cython 
and Rust together in a high-performance and efficient fashion. Aimed at replacing the core analysis 
functionality found in Pandas; only _really_ fast and memory efficient. 


*This package is in Alpha and in no-way can you expect this to be functional or reliable.*

### Project outlook *(...the long story)*:

The _(long term)_ goal for this project is to provide a light-weight alternative to
the fantastic `pandas`. I love and use pandas all the time, so this is what has 
inspired me to making something similar; but excelling in a few areas such as minimizing
memory footprint and speeding up certain computations via Rust's speed and safety 
guarantees.  

This project shall have the same concept of `dataframe` & `series`; these objects 
will be stored as Rust structures. When "displayed" in Python
it will merely be some meta-data description of the vector. Most computations will
take place there, those which can't or perhaps better implemented in numpy will continue 
to be done in numpy (via pointer transferals, thus free of copies).

The point is, I'm striving to practice efficiency, parallelism, safety & speed with this 
project while maintaining some of the most valuable functionality of pandas. 

---

### Install checklist

- [Rustup](https://rustup.rs/)
    - `rustup install nightly`
    - `rustup default nightly`
- gcc >= 7.x.x 
- g++ >= 7.x.x


**NOTE** Only Python 3.5 is being tested against on Unix platforms


#####Installing from command line:
```commandline
# Clone repo:
git clone https://github.com/milesgranger/lumber-jack.git && cd lumber-jack

# Run tests
LD_LIBRARY_PATH=$(pwd)/lumberjack/rust:$LD_LIBRARY_PATH python setup.py test

# Install
python setup.py build_ext && python setup.py install

# Uninstall
pip uninstall lumber-jack
```
---

### Working Pandas & Numpy like operations:

##### There isn't much, but check back soon! ;)
```python
import lumberjack as lj

# lj.Series is a drop-in replacement for pandas.Series, overriding these methods:
series = lj.Series.arange(0, 10000)  # ~8x  faster than numpy
series.sum()                         # ~40x faster than pandas & ~3x  faster than numpy
series.cumsum()                      # ~4x  faster than pandas & ~0x  faster than numpy
series.mean()                        # ~98x faster than pandas & ~20x faster than numpy
series * 2                           # ~8x  faster than pandas & ~0x  faster than numpy
series *= 2                          # ~10x faster than pandas & ~0x  faster than numpy
series + 2                           # ....
series += 2                          # ....

```

### Working Alteration Tools:

Split arrays of text which are separated by < something > into a one-hot encoded ndarray

```python
from lumberjack import alterations
raw_texts = ['hello, there', 'hi, there']
alterations.split_n_one_hot_encode(raw_texts, sep=',', cutoff=0)
(['hello', 'there', 'hi'], [[1, 1, 0], [0, 1, 1]])
```
