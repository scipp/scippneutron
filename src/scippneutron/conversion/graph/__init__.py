# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

# flake8: noqa[F401]
"""Graphs for coordinate transformations.

All graphs are defined in terms of the functions in the parent module
:mod:`scippneutron.conversion`.
See there for definitions of the individual conversions.

Typically, multiple graphs need to be combined for a full transformation.
For example:

    >>> from scippneutron.conversion import graph
    >>> wavelength_graph = {**graph.beamline.beamline(scatter=True),
    ...                     **graph.tof.elastic_wavelength(start='tof')}

The `user guide <../../user-guide/coordinate-transformations.rst>`_ gives
more examples.
"""

from . import beamline
from . import tof
