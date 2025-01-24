# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Metadata utilities.

.. rubric:: Models

.. autosummary::
  :toctree: ../classes
  :template: model-template.rst

  Beamline
  Measurement
  Person
  Software
  Source

.. rubric:: Auxiliary Classes

.. autosummary::
  :toctree: ../classes
  :template: class-template.rst

  ORCIDiD
  RadiationProbe
  SourceType

.. rubric:: Attributes

.. autosummary::

  ESS_SOURCE
"""

from ._model import (
    Beamline,
    Person,
    Software,
    Source,
    Measurement,
    SourceType,
    ESS_SOURCE,
    RadiationProbe,
)
from ._orcid import ORCIDiD

__all__ = [
    'Beamline',
    'Measurement',
    'Person',
    'ORCIDiD',
    'Software',
    'Source',
    'SourceType',
    'ESS_SOURCE',
    'RadiationProbe',
]
