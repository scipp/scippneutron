#!/bin/bash

set -ex

if test -z "${INSTALL_PREFIX}"
then
  export INSTALL_PREFIX="$(pwd)/scippneutron"
  ./tools/make_and_install.sh
fi

mv "$INSTALL_PREFIX"/scippneutron "$CONDA_PREFIX"/lib/python*/
mv "$INSTALL_PREFIX"/lib/libscippneutron* "$CONDA_PREFIX"/lib/
mv "$INSTALL_PREFIX"/lib/cmake/scippneutron "$CONDA_PREFIX"/lib/cmake/
mv "$INSTALL_PREFIX"/include/scippneutron* "$CONDA_PREFIX"/include/
mv "$INSTALL_PREFIX"/include/scipp "$CONDA_PREFIX"/include/
