# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Models for peaks and background."""

import abc
import math
from typing import Iterable

import numpy as np
import scipp as sc


class Model(abc.ABC):
    def __init__(self, *, prefix: str, param_names: Iterable[str]) -> None:
        self._prefix = prefix
        self._param_names = set(param_names)

    @abc.abstractmethod
    def _call(self, x: sc.Variable, params: dict[str, sc.Variable]) -> sc.Variable:
        ...

    @abc.abstractmethod
    def _guess(self, x: sc.Variable, y: sc.Variable) -> dict[str, sc.Variable]:
        ...

    def __call__(self, x: sc.Variable, **params: sc.Variable) -> sc.Variable:
        if params.keys() != self._param_names:
            raise ValueError(
                f'Bad parameters for model {self.__class__.__name__},'
                f'got: {set(params.keys())}, expected {self._param_names}'
            )
        return self._call(x, params)

    def guess(
        self, data: sc.DataArray, *, coord: str | None = None
    ) -> dict[str, sc.Variable]:
        if coord is None:
            coord = data.dim
        return {
            self._prefix + name: param
            for name, param in self._guess(x=data.coords[coord], y=data.data).items()
        }


class LinearModel(Model):
    def __init__(self, *, prefix: str) -> None:
        super().__init__(prefix=prefix, param_names=('slope', 'offset'))

    def _call(self, x: sc.Variable, params: dict[str, sc.Variable]) -> sc.Variable:
        slope = params['slope']
        offset = params['offset']

        val = slope * x
        val += offset
        return val

    def _guess(self, x: sc.Variable, y: sc.Variable) -> dict[str, sc.Variable]:
        poly = np.polynomial.Polynomial.fit(x.values, y.values, deg=1)
        offset, slope = poly.convert().coef
        return {
            'offset': sc.scalar(offset, unit=y.unit),
            'slope': sc.scalar(slope, unit=y.unit / x.unit),
        }


class GaussianModel(Model):
    def __init__(self, *, prefix: str) -> None:
        super().__init__(prefix=prefix, param_names=('amplitude', 'loc', 'width'))

    def _call(self, x: sc.Variable, params: dict[str, sc.Variable]) -> sc.Variable:
        amplitude = params['amplitude']
        loc = params['loc']
        scale = params['scale']

        val = (x - loc) ** 2
        val /= 2 * scale**2
        val = sc.exp(val, out=val)
        val *= amplitude / (math.sqrt(2 * math.pi) * scale)
        return val

    def _guess(self, x: sc.Variable, y: sc.Variable) -> dict[str, sc.Variable]:
        return {name: param for name, param in _guess_from_peak(x, y).items()}


def _guess_from_peak(x: sc.Variable, y: sc.Variable) -> dict[str, sc.Variable]:
    """Estimate the parameters of a peaked function.

    The estimation is based on a Gaussian but
    is good enough for similar functions, too.

    The function was adapted from ``lmfit.models.guess_from_peak``, see
    https://github.com/lmfit/lmfit-py/blob/e57aab2fe2059efc07535a67e4fdc577291e9067/lmfit/models.py#L42
    """
    y_min, y_max = sc.min(y), sc.max(y)

    # These are the points within FWHM of the peak.
    x_half_max = x[y > (y_max + y_min) / 2.0]
    if len(x_half_max) > 2:
        loc = x_half_max.mean()
        # Rough estimate of sigma ~ FWHM / 2.355 for Gaussian.
        scale = (x_half_max.max() - x_half_max.min()) / 2.0
    else:
        loc = x[np.argmax(y.values)]
        # 6.0 taken from lmfit, don't know where it comes from.
        scale = (sc.max(x) - sc.min(x)) / 6.0

    # 3.0 approximates sqrt(2*pi) in the normalization of a Gaussian.
    height = (y_max - y_min) * 3.0
    amplitude = height * scale
    # TODO lower bound for sigma
    #   also for amplitude in the actual peak fit but not necessarily here
    return {
        'amplitude': amplitude,
        'loc': loc,
        'scale': scale,
    }
