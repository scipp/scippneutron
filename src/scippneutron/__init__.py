# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401, I

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

del importlib

import lazy_loader as lazy

submodules = [
    'absorption',
    'atoms',
    'chopper',
    'io',
    'conversion',
    'data',
    'metadata',
    'peaks',
    'tof',
]

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=submodules,
    submod_attrs={
        'beamline_components': [
            'position',
            'source_position',
            'sample_position',
            'incident_beam',
            'scattered_beam',
            'Ltotal',
            'L1',
            'L2',
            'two_theta',
        ],
        'core': ['convert', 'conversion_graph', 'deduce_conversion_graph'],
        'mantid': ['from_mantid', 'to_mantid', 'load_with_mantid', 'fit'],
        'instrument_view': ['instrument_view'],
        'masking': ['MaskingTool'],
    },
)
