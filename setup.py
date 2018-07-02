
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from lumberjack._version import __version__

try:
    from setuptools_rust import RustExtension
except ImportError:
    import subprocess
    errno = subprocess.call([sys.executable, '-m', 'pip', 'install', 'setuptools-rust>=0.9.2'])
    if errno:
        print("Please install setuptools-rust package")
        raise SystemExit(errno)
    else:
        from setuptools_rust import RustExtension

setup_requires = ['setuptools-rust>=0.9.2', 'pytest-runner', "Cython"]
install_requires = ['numpy', 'pandas']
tests_require = install_requires + ['pytest==3.5.0', 'pytest-benchmark']

rust_core_ext = Extension(name="*",
                          sources=["lumberjack/cython/*.pyx"],
                          libraries=['lumberjacklib'],
                          extra_link_args=['-L./lumberjack/rust'],
                          library_dirs=['./lumberjack/rust'])

setup(
    name='lumber-jack',
    version=__version__,
    maintainer="Miles Granger",
    keywords="pandas rust python data manipulation processing",
    url="https://github.com/milesgranger/lumber-jack",
    description="Alpha work on adding additional and improved functionality to Pandas.",
    long_description="Alpha work on adding additional and improved functionality to Pandas.",
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Rust',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS :: MacOS X',
    ],
    packages=['lumberjack'],
    include_dirs=[np.get_include()],
    ext_modules=cythonize([rust_core_ext], include_path=[np.get_include()]),
    rust_extensions=[
        RustExtension('lumberjack.rust.lumberjacklib', 'Cargo.toml')
    ],
    install_requires=install_requires,
    tests_require=tests_require,
    test_suite='tests',
    setup_requires=setup_requires + install_requires,
    include_package_data=True,
    license="OSI Approved :: BSD License",
    zip_safe=False
)

