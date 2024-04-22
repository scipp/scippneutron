# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""Peak fitting and removal.

.. rubric:: Classes

.. autosummary::
  :toctree:
  :template: class-template.rst

  FitAssessment
  FitResult

.. rubric:: Functions

.. autosummary::
  :toctree:

  assess_fit
  fit_peaks
  remove_peaks
"""

from ._common import FitParameters, FitRequirements
from ._fit_peaks import FitAssessment, FitResult, assess_fit, fit_peaks
from ._remove_peaks import remove_peaks

__all__ = [
    'FitAssessment',
    'FitParameters',
    'FitRequirements',
    'FitResult',
    'assess_fit',
    'fit_peaks',
    'remove_peaks',
]
