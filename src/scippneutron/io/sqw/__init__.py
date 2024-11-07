# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

from ._sqw import Sqw

from ._build import SqwBuilder
from ._bytes import Byteorder
from ._models import (
    SqwFileType,
    SqwFileHeader,
    SqwMainHeader,
    EnergyMode,
    SqwIXExperiment,
    SqwDndMetadata,
    SqwLineAxes,
    SqwLineProj,
    SqwIXNullInstrument,
    SqwIXSample,
    SqwIXSource,
)

__all__ = [
    "Byteorder",
    "EnergyMode",
    "SqwIXExperiment",
    "SqwMainHeader",
    "Sqw",
    "SqwBuilder",
    "SqwFileType",
    "SqwFileHeader",
    "SqwDndMetadata",
    "SqwLineAxes",
    "SqwLineProj",
    "SqwIXNullInstrument",
    "SqwIXSample",
    "SqwIXSource",
]
