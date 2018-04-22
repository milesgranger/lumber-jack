
import sys
import lumberjack
from setuptools import setup

try:
    from setuptools_rust import RustExtension
except ImportError:
    import subprocess
    errno = subprocess.call([sys.executable, '-m', 'pip', 'install', 'setuptools-rust>=0.9.1'])
    if errno:
        print("Please install setuptools-rust package")
        raise SystemExit(errno)
    else:
        from setuptools_rust import RustExtension


setup_requires = ['setuptools-rust>=0.9.1']
install_requires = ['numpy']
tests_require = install_requires + ['pytest', 'pytest-benchmark']

setup(
    name='lumber-jack',
    version=lumberjack.__version__,
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
    rust_extensions=[
        RustExtension('lumberjack.rust.alterations', 'Cargo.toml')
    ],
    install_requires=install_requires,
    tests_require=tests_require,
    test_suite='lumberjack.tests',
    setup_requires=setup_requires,
    include_package_data=True,
    license="OSI Approved :: BSD License",
    zip_safe=False
)

