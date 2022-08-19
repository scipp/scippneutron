# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
"""
Internal utilities; do not use outside scippneutron!
"""

from scipp.typing import VariableLike
import scipp as sc


def elem_unit(var: VariableLike) -> sc.Unit:
    return var.bins.unit if var.bins is not None else var.unit


def elem_dtype(var: VariableLike) -> sc.DType:
    return var.bins.constituents['data'].dtype if var.bins is not None else var.dtype


def float_dtype(var: VariableLike) -> sc.DType:
    dtype = elem_dtype(var)
    if dtype == sc.DType.float32:
        return sc.DType.float32
    return sc.DType.float64


def as_float_type(var: VariableLike, ref: VariableLike) -> VariableLike:
    return var.astype(float_dtype(ref), copy=False)
