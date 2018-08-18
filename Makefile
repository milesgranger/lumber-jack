

VERSION := 0.0.1

build-docs:
	cargo doc --no-deps --open

test-python:
	LD_LIBRARY_PATH=$(shell pwd)/lumberjack/rust python setup.py test

test-rust:
	cargo test