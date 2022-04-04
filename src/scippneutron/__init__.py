# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
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

from .core import __version__
from .core import position, source_position, sample_position, incident_beam, scattered_beam, Ltotal, L1, L2, two_theta
from .core import convert
from .mantid import from_mantid, to_mantid, load, fit
from .instrument_view import instrument_view
from .file_loading.load_nexus import load_nexus, load_nexus_json
from .data_streaming.data_stream import data_stream
from . import data
