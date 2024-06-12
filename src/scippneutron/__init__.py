# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402

"""Neutron scattering toolkit built using scipp for Data Reduction.

ScippNeutron is a generic (as in 'usable by different facilities')
package for data processing in neutron scattering.
It provides coordinate transformations, file I/O, and technique-specific tools.

See the online documentation for user guides and the API reference:
https://scipp.github.io/scippneutron/
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from .beamline_components import (
    position,
    source_position,
    sample_position,
    incident_beam,
    scattered_beam,
    Ltotal,
    L1,
    L2,
    two_theta,
)
from .core import convert
from .mantid import (
    from_mantid,
    array_from_mantid,
    to_mantid,
    load_with_mantid,
    load,
    fit,
)
from .instrument_view import instrument_view
from .io.nexus.load_nexus import load_nexus, load_nexus_json
from .data_streaming.data_stream import data_stream
from .masking import MaskingTool
from . import atoms
from . import data
from . import chopper

del importlib

__all__ = [
    "position",
    "source_position",
    "sample_position",
    "incident_beam",
    "scattered_beam",
    "Ltotal",
    "L1",
    "L2",
    "two_theta",
    "convert",
    "from_mantid",
    "array_from_mantid",
    "to_mantid",
    "load_with_mantid",
    "load",
    "fit",
    "instrument_view",
    "load_nexus",
    "load_nexus_json",
    "data_stream",
    "atoms",
    "chopper",
    "data",
    "MaskingTool",
]
