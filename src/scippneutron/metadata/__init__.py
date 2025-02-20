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
    ESS_SOURCE,
    Beamline,
    Measurement,
    Person,
    RadiationProbe,
    Software,
    Source,
    SourceType,
)
from ._orcid import ORCIDiD

__all__ = [
    'ESS_SOURCE',
    'Beamline',
    'Measurement',
    'ORCIDiD',
    'Person',
    'RadiationProbe',
    'Software',
    'Source',
    'SourceType',
]
