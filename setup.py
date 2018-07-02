from subprocess import Popen
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np
from lumberjack._version import __version__

setup_requires = ['pytest-runner', "Cython"]
install_requires = ['numpy']
tests_require = install_requires + ['pytest==3.5.0', 'pytest-benchmark']


class BuildRustLib(build_ext):
    """
    Augment the setup.py build_ext command to creat the Rust shared object
    """
    def run(self):
        print('running build of liblumberjack')
        process = Popen(['cargo', 'build', '--out-dir', './lumberjack/rust/', '-Z', 'unstable-options', '--release'])
        process.wait()
        if process.returncode != 0:
            raise RuntimeError('Failed building liblumberjack object, exit code: {}'.format(process.returncode))
        super().run()


rust_core_ext = Extension(name="*",
                          sources=["lumberjack/cython/*.pyx"],
                          libraries=['lumberjack'],
                          include_dirs=['./lumberjack/rust/', np.get_include()],
                          library_dirs=['./lumberjack/rust/'])

setup(
    name='lumber-jack',
    version=__version__,
    maintainer="Miles Granger",
    keywords="pandas rust python data manipulation processing",
    url="https://github.com/milesgranger/lumber-jack",
    description="Alpha work on adding additional and improved functionality to Pandas.",
    long_description="Alpha work on adding additional and improved functionality to Pandas.",
    cmdclass={
      'build_ext': BuildRustLib
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Rust',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS :: MacOS X',
    ],
    packages=find_packages(),
    include_dirs=[np.get_include(), 'lumberjack/rust/'],
    ext_modules=cythonize([rust_core_ext]),
    install_requires=install_requires,
    tests_require=tests_require,
    test_suite='tests',
    setup_requires=setup_requires + install_requires,
    include_package_data=True,
    license="OSI Approved :: BSD License",
    zip_safe=False
)

