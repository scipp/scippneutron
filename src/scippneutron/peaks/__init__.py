# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""Peak fitting and removal.

.. rubric:: Classes

.. autosummary::
  :toctree:
  :template: class-template.rst

  FitAssessment
  FitParameters
  FitResult
  FitRequirements

.. rubric:: Functions

.. autosummary::
  :toctree:

  fit_peaks
  remove_peaks
"""

from ._common import FitParameters, FitRequirements
from ._fit_peaks import FitAssessment, FitResult, fit_peaks
from ._remove_peaks import remove_peaks

__all__ = [
    'FitAssessment',
    'FitParameters',
    'FitRequirements',
    'FitResult',
    'fit_peaks',
    'remove_peaks',
]
