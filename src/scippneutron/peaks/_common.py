# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class FitRequirements:
    min_p_value: float = 0.01
    max_peak_width_factor: float = 1.0
    min_peak_width_factor: float = 1.0


@dataclass(kw_only=True, slots=True)
class FitParameters:
    guess_background_range: int = 2
    neighbor_separation_factor: float = 1 / 3
