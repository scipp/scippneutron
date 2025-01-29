# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

"""Components for coordinate transformations.

The submodules are mainly intended for constructing graphs for
:func:`scipp.transform_coords` but can also be used independently.

``beamline`` and ``tof`` contain individual transformation kernels.
``graph`` contains concrete graphs that can be passed to ``transform_coords``.

See also the user guide on
`Coordinate Transformations <../../user-guide/coordinate-transformations.rst>`_.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
