# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Iterable

import scipp as sc

from ._fit_peaks import FitResult


def remove_peaks(data: sc.DataArray, fit_results: Iterable[FitResult]) -> sc.DataArray:
    """Remove peaks from data by subtracting fitted models.

    Peaks are removed by subtracting fitted peak models from the data.
    The background models are ignored in this operation.

    Parameters
    ----------
    data:
        1d data with peaks.
        Must have a dimension-coordinate.
    fit_results:
        Results of fitting the peaks that should be removed.
        Any unsuccessful fits are ignored.

    Returns
    -------
    :
        ``data`` with peaks removed.

    See Also
    --------
    fit_peaks:
        Fit the peaks in ``data`` and construct the required input to ``fit_results``.
    """
    if data.variances is not None:
        raise sc.VariancesError(
            'Cannot remove peaks from data with variances as that would introduce '
            'correlations between data points.'
        )

    data = data.copy(deep=False)
    data.data = data.data.copy()  # need a deep copy for the in-place subtraction below
    for result in fit_results:
        if not result.success:
            continue
        in_window = data[data.dim, result.window[0] : result.window[1]]
        in_window -= result.eval_peak(in_window.coords[data.dim])
    return data
