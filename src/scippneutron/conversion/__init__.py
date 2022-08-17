# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

# flake8: noqa[F401]
"""Components for coordinate transformations.

The submodules are mainly intended for constructing graphs for
:func:`scipp.transform_coords` but can also be used independently.

``beamline`` and ``tof`` contain individual transformation kernels.
``graph`` contains concrete graphs that can be passed to ``transform_coords``.

See also the user guide on
`Coordinate Transformations <../../user-guide/coordinate-transformations.rst>`_.
"""

from . import beamline
from . import tof
