# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
"""
Internal utilities; do not use outside scippneutron!
"""

from collections.abc import MutableMapping

import scipp as sc
from scipp.typing import VariableLike


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


def get_attrs(da: sc.DataArray) -> MutableMapping[str, sc.Variable]:
    try:
        # During deprecation phase
        return da.deprecated_attrs
    except AttributeError:
        try:
            # Before deprecation phase
            return da.attrs
        except AttributeError:
            # After deprecation phase / removal of attrs
            return da.coords


def get_meta(da: sc.DataArray) -> MutableMapping[str, sc.Variable]:
    try:
        # During deprecation phase
        return da.deprecated_meta
    except AttributeError:
        try:
            # Before deprecation phase
            return da.meta
        except AttributeError:
            # After deprecation phase / removal of attrs
            return da.coords
