#!/bin/bash
set -e -x

# build wheels locally: docker run --rm -v $(pwd):/io quay.io/pypa/manylinux1_x86_64 bash /io/travis/build-wheels.sh

# Install rust and set to nightly dist
curl https://sh.rustup.rs -sSf > rust-init.sh && bash rust-init.sh -y
export PATH=$PATH:$HOME/.cargo/bin
rustup default nightly

# Install a system package required by our library
yum install -y atlas-devel

# Compile wheels
cd /io
for PYBIN in /opt/python/*/bin; do
    echo "Using python: ${PYBIN}"
    {
        "${PYBIN}/pip" install --upgrade pip wheel setuptools
        "${PYBIN}/pip" install -r /io/requirements.txt
        "${PYBIN}/python" setup.py build_ext
        "${PYBIN}/pip" wheel . -w /io/wheelhouse
    } || {
        echo "Failed to build wheel using ${PYBIN}"
    }
done

# Bundle external shared libraries into the wheels
for whl in /io/wheelhouse/lumber_jack*.whl; do
    auditwheel repair "$whl" -w /io/dist/
done


# Install packages and test
#for PYBIN in /opt/python/*/bin/; do
#    "${PYBIN}/pip" install lumber-jack --no-index -f /io/dist
#    (cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
#done

exit 0