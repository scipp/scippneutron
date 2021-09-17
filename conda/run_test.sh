#!/bin/bash

cd docs
docs_build_dir=$(mktemp -d)
python build_docs.py --builder=html --prefix=$docs_build_dir --work_dir=$(mktemp -d)
python build_docs.py --builder=doctest --prefix=$docs_build_dir --work_dir=$(mktemp -d) --no-setup
