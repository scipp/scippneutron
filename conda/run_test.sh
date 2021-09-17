#!/bin/bash

cd cmake-package-test
mkdir -p build && cd build
cmake -DPKG_VERSION=$PKG_VERSION -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
cd ..

cd docs
docs_build_dir=$(mktemp -d)
python build_docs.py --builder=html --prefix=$docs_build_dir --work_dir=$(mktemp -d)
python build_docs.py --builder=doctest --prefix=$docs_build_dir --work_dir=$(mktemp -d) --no-setup
cd ..
