# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import Iterable

import scipp as sc

from ._fit_peaks import FitResult


def remove_peaks(data: sc.DataArray, fit_results: Iterable[FitResult]) -> sc.DataArray:
    data = data.copy(deep=False)
    for result in fit_results:
        if not result.success:
            continue
        in_window = data[data.dim, result.window[0] : result.window[1]]
        in_window -= result.eval_peak(in_window.coords[data.dim])
    return data
