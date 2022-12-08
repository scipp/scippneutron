# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

# flake8: noqa

import importlib.metadata
try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from .beamline_components import position, source_position, sample_position, incident_beam, scattered_beam, Ltotal, L1, L2, two_theta
from .core import convert
from .mantid import from_mantid, to_mantid, load, fit
from .instrument_view import instrument_view
from .file_loading.load_nexus import load_nexus, load_nexus_json
from .data_streaming.data_stream import data_stream
from . import data
