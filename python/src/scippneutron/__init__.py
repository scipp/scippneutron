# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @file
# @author Simon Heybrock

# flake8: noqa

import os

if os.name == "nt":
    import importlib.resources
    with importlib.resources.path("scipp", "__init__.py") as path:
        # For scipp libs
        os.environ["PATH"] += os.pathsep + str(path.parent.resolve())
        # For TBB
        os.environ["PATH"] += os.pathsep + str(
            (path.parent.parent.parent / "Library" / "Bin").resolve())

    with importlib.resources.path("scippneutron", "__init__.py") as path:
        # For scippneutron lib
        os.environ["PATH"] += os.pathsep + str((path.parent).resolve())

from ._scippneutron import __version__
from ._scippneutron import convert
from ._scippneutron import position, source_position, sample_position, incident_beam, scattered_beam, Ltotal, L1, L2, two_theta
from .mantid import from_mantid, to_mantid, load, fit
from .instrument_view import instrument_view
from .file_loading.load_nexus import load_nexus, load_nexus_json
from .data_streaming.data_stream import data_stream
