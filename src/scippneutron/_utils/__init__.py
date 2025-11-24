# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
"""
Internal utilities; do not use outside scippneutron!
"""

from typing import TypeVar

import scipp as sc

_V = TypeVar('_V', sc.Variable, sc.DataArray)


def elem_unit(var: sc.Variable | sc.DataArray) -> sc.Unit:
    unit = var.bins.unit if var.is_binned else var.unit
    if unit is None:
        raise sc.UnitError("Cannot do arithmetic with variables without units")
    return unit


def elem_dtype(var: sc.Variable | sc.DataArray) -> sc.DType:
    return var.bins.constituents['data'].dtype if var.is_binned else var.dtype  # type: ignore[union-attr]


def float_dtype(var: sc.Variable | sc.DataArray) -> sc.DType:
    dtype = elem_dtype(var)
    if dtype == sc.DType.float32:
        return sc.DType.float32
    return sc.DType.float64


def as_float_type(var: _V, ref: sc.Variable | sc.DataArray) -> _V:
    return var.astype(float_dtype(ref), copy=False)
