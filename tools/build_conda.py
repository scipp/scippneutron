# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

import os
import shutil
import glob
import sys
import build_cpp
import scippbuildtools as sbt

if __name__ == '__main__':

    # Search for a defined SCIPP_INSTALL_PREFIX env variable.
    # If it exists, it points to a previously built target and we simply move
    # the files from there into the conda build directory.
    # If it is undefined, we build the C++ library by calling main() from
    # build_cpp.
    source_root = os.environ.get('SCIPP_INSTALL_PREFIX')
    if source_root is None:
        source_root = os.path.abspath('scippneutron_install')
        build_cpp.main(prefix=source_root)
    destination_root = os.environ.get('CONDA_PREFIX')

    # Create a file mover to place the built files in the correct directories
    # for conda build.
    m = sbt.FileMover(source_root=source_root, destination_root=destination_root)

    # Depending on the platform, directories have different names.
    if sys.platform == "win32":
        lib_dest = 'lib'
        bin_src = 'bin'
        lib_src = 'Lib'
        inc_src = 'include'
    else:
        lib_dest = os.path.join('lib', 'python*')
        bin_src = None
        lib_src = 'lib'
        inc_src = 'include'

    m.move(['scippneutron'], [lib_dest])
    if bin_src is not None:
        m.move([bin_src, 'scippneutron*.dll'], [bin_src])
    m.move([lib_src, '*scippneutron*'], [lib_src])
    m.move([lib_src, 'cmake', 'scippneutron'], [lib_src, 'cmake'])
    m.move([inc_src, 'scippneutron*'], [inc_src])
    m.move([inc_src, 'scipp', 'neutron'], [inc_src, 'scipp'])
