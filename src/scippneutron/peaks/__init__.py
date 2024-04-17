# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""Peak fitting and removal."""

from ._fit_peaks import FitAssessment, FitResult, assess_fit, fit_peaks
from ._remove_peaks import remove_peaks

__all__ = ['FitAssessment', 'FitResult', 'assess_fit', 'fit_peaks', 'remove_peaks']
