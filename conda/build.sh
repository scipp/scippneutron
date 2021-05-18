#!/bin/bash

set -ex

# conda-build breaks all of the key mechanisms used by ccache,
# preventing a cache hit from ever happening.
# To allow the use of ccache in CI builds, we call an external build script
# (called make_and_install.sh) first to build the C++ library,
# run the C++ tests, and build the python library.
# In this case, the INSTALL_PREFIX will be defined.
# If it is not defined (e.g. for a local conda-build), we need to call that
# same script here.
# Using this separate script also minimizes code duplication.

if test -z "${INSTALL_PREFIX}"
then
  export INSTALL_PREFIX="$(pwd)/scippneutron_install"
  ./tools/make_and_install.sh
fi

# The final step in the conda-build process is to move the install targets to
# the correct location to build the package.

mv "$INSTALL_PREFIX"/scippneutron "$CONDA_PREFIX"/lib/python*/
mv "$INSTALL_PREFIX"/lib/libscippneutron* "$CONDA_PREFIX"/lib/
mv "$INSTALL_PREFIX"/lib/cmake/scippneutron "$CONDA_PREFIX"/lib/cmake/
mv "$INSTALL_PREFIX"/include/scippneutron* "$CONDA_PREFIX"/include/
mv "$INSTALL_PREFIX"/include/scipp/* "$CONDA_PREFIX"/include/scipp/
