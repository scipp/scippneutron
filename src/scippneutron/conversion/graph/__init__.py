# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

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

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
