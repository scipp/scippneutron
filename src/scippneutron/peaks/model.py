# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Models for peaks and background."""

from __future__ import annotations

import abc
import math
from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import scipp as sc


class Model(abc.ABC):
    """Abstract base class for fitting models.

    This class defines the basic interface for models by way of public methods.
    Subclasses should override the protected methods ``_call``, ``_guess``, and
    optionally ``_param_bounds`` instead of their public counterparts.
    """

    def __init__(self, *, param_names: Iterable[str], prefix: str = '') -> None:
        """Initialize a base model.

        Parameters
        ----------
        param_names:
            Names of parameters in arbitrary order.
            Does not include the prefix.
        prefix:
            Prefix used for model parameters in all user-facing data.
        """
        self._prefix = prefix
        self._param_names = set(param_names)
        self._prefixed_param_names = {prefix + name for name in self._param_names}

    @abc.abstractmethod
    def _call(self, x: sc.Variable, params: dict[str, sc.Variable]) -> sc.Variable:
        """Evaluate the model at a given independent variable and parameters.

        Parameters
        ----------
        x:
            Independent variable.
        params:
            Dict from parameter names to values.
            Names are given *without* prefix.

        Returns
        -------
        :
            Model evaluated at the given independent variable.
        """
        ...

    @abc.abstractmethod
    def _guess(self, x: sc.Variable, y: sc.Variable) -> dict[str, sc.Variable]:
        """Roughly estimate the model parameters.

        Parameters
        ----------
        x:
            Independent variable.
        y:
            Dependent variable.

        Returns
        -------
        :
            Estimated parameters.
            Dict keys are parameter names *without* the prefix.
        """
        ...

    def _param_bounds(self) -> dict[str, tuple[float, float]]:
        """Return bounds for parameters.

        Returns
        -------
        :
            Upper and lower bounds for parameters.
            Dict keys are parameter names *without* the prefix.
            Parameters omitted from this dict are unbounded.
        """
        return {}

    @property
    def prefix(self) -> str:
        """Prefix for parameter names."""
        return self._prefix

    @property
    def param_names(self) -> set[str]:
        """Parameter names including the prefix."""
        return self._prefixed_param_names.copy()

    def __call__(self, x: sc.Variable, **params: sc.Variable) -> sc.Variable:
        """Evaluate the model.

        Parameters
        ----------
        x:
            Independent variable.
        params:
            Parameter values.

        Returns
        -------
        :
            Model evaluated at the given independent variable and parameters.
        """
        if params.keys() != self.param_names:
            raise ValueError(
                f'Bad parameters for model {self.__class__.__name__}, '
                f'got: {set(params.keys())}, expected {self.param_names}'
            )
        return self._call(
            x, {name[len(self._prefix) :]: val for name, val in params.items()}
        )

    def guess(
        self, data: sc.DataArray, *, coord: str | None = None
    ) -> dict[str, sc.Variable]:
        """Roughly estimate the model parameters for given data.

        The estimate can be used as the starting point for a fit
        but does not necessarily represent a good fit by itself.

        Parameters
        ----------
        data:
            Data array where ``data.data`` is the dependent variable
            and a chosen coord (see below) is the independent variable.
        coord:
            Coordinate name of ``data`` to use as independent variable.
            If not given, ``data.dim`` is used instead.

        Returns
        -------
        :
            Estimated parameters.
        """
        if coord is None:
            coord = data.dim
        return {
            self._prefix + name: param
            for name, param in self._guess(x=data.coords[coord], y=data.data).items()
        }

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        """Parameter bounds.

        Returns
        -------
        :
            Upper and lower bounds for parameters.
            Parameters omitted from this dict are unbounded.
        """
        return {
            self._prefix + name: bounds for name, bounds in self._param_bounds().items()
        }

    def fwhm(self, params: dict[str, sc.Variable]) -> sc.Variable:
        """Compute full width at half maximum.

        Note that this function is only implemented for peaked models!

        Parameters
        ----------
        params:
            Parameter values for which to compute the FWHM.

        Returns
        -------
        :
            FWHM of the model at the given parameters.

        Raises
        ------
        NotImplementedError:
            If this model does not support computing the FWHM.
        """
        raise NotImplementedError(
            f'FWHM is not implemented for model {self.__class__.__name__}'
        )

    def __add__(self, other: Model) -> CompositeModel:
        """Combine two models into a :class:`CompositeModel`."""
        if not isinstance(other, Model):
            return NotImplemented
        return CompositeModel(left=self, right=other, prefix='')

    def with_prefix(self, prefix: str) -> Model:
        """Return a copy of the model with a new prefix."""
        model = deepcopy(self)
        model._prefix = prefix
        model._prefixed_param_names = {prefix + name for name in model._param_names}
        return model


class CompositeModel(Model):
    """A combination of two models.

    Composite models contain a "left" and a "right" submodel which are combined into

    .. math::

        f(x) = \\text{left}(x) + \\text{right}(x)

    Composite models can be constructed by adding models, e.g.,

    .. code-block:: python

        left = PolynomialModel(degree=2)
        right = GaussianModel()
        composite = left + right

    The parameters of the composite are the union of the component parameters.
    If there is a clash between the names of component models, they must
    be disambiguated by using prefixes.
    """

    def __init__(self, left: Model, right: Model, *, prefix: str = '') -> None:
        """Initialize a composite model.

        Parameters
        ----------
        left:
            Left component model.
        right:
            Right component model.
        prefix:
            Prefix for *all* model parameter names.
            It is prepended to the prefixes of the component models.
        """
        if left.param_names & right.param_names:
            raise ValueError(
                f'Model {left.__class__.__name__} and model {right.__class__.__name__} '
                'have overlapping parameter names: '
                f'{left.param_names & right.param_names}. '
                'Use prefixes to disambiguate.'
            )
        self._left = left
        self._right = right
        super().__init__(
            prefix=prefix, param_names=left.param_names | right.param_names
        )

    def _call(self, x: sc.Variable, params: dict[str, sc.Variable]) -> sc.Variable:
        left = self._left(x, **{name: params[name] for name in self._left.param_names})
        right = self._right(
            x, **{name: params[name] for name in self._right.param_names}
        )
        return left + right

    def _guess(self, x: sc.Variable, y: sc.Variable) -> dict[str, sc.Variable]:
        data = sc.DataArray(y, coords={y.dim: x})
        return {
            **self._left.guess(data),
            **self._right.guess(data),
        }

    def _param_bounds(self) -> dict[str, tuple[float, float]]:
        return self._left.param_bounds | self._right.param_bounds


class PolynomialModel(Model):
    """A polynomial of fixed degree.

    ``PolynomialModel(degree=n)`` implements

    .. math::

        f(x; a_0, \\ldots, a_n) = \\sum_{i=0}^{n}\\,a_i x^i

    where the sum is inclusive on the upper bound.
    :math:`a_i` are the parameters and are named ``['a0', 'a1', ...]``.
    """

    def __init__(self, *, degree: int, prefix: str = '') -> None:
        """Initialize a polynomial model.

        Parameters
        ----------
        degree:
            Degree of the polynomial.
        prefix:
            Prefix for model parameter names.
        """
        if degree <= 0:
            raise ValueError(f'Degree must be positive, got: {degree}')
        super().__init__(
            prefix=prefix, param_names=(f'a{i}' for i in range(degree + 1))
        )

    @property
    def degree(self) -> int:
        """The degree of the polynomial."""
        return len(self._param_names) - 1

    def _call(self, x: sc.Variable, params: dict[str, sc.Variable]) -> sc.Variable:
        a_degree = params[f'a{self.degree}']
        val = sc.full(value=a_degree.value, unit=a_degree.unit, sizes=x.sizes)
        for i in range(self.degree - 1, -1, -1):
            val *= x
            val += params[f'a{i}']
        return val

    def _guess(self, x: sc.Variable, y: sc.Variable) -> dict[str, sc.Variable]:
        poly = np.polynomial.Polynomial.fit(x.values, y.values, deg=self.degree)
        return {
            f'a{i}': sc.scalar(c, unit=y.unit / x.unit**i)
            for i, c in enumerate(poly.convert().coef)
        }


class GaussianModel(Model):
    r"""A Gaussian function with arbitrary normalization.

    The model implements

    .. math::

        f(x; A, \mu, \sigma) = \frac{A}{\sqrt{2\pi}\sigma}
          \exp{\left(-\frac{{(x-\mu)}^2}{2\sigma^2}\right)}

    with parameters

    - :math:`A`: ``'amplitude'``
    - :math:`\mu`: ``'loc'``
    - :math:`\sigma`: ``'scale'``
    """

    def __init__(self, *, prefix: str = '') -> None:
        """Initialize a Gaussian model.

        Parameters
        ----------
        prefix:
            Prefix for model parameter names.
        """
        super().__init__(prefix=prefix, param_names=('amplitude', 'loc', 'scale'))

    def _call(self, x: sc.Variable, params: dict[str, sc.Variable]) -> sc.Variable:
        return _gaussian(x, **params)

    def _guess(self, x: sc.Variable, y: sc.Variable) -> dict[str, sc.Variable]:
        params = _guess_from_peak(x, y)
        # Adjust by the normalization factor of gaussians.
        params['amplitude'] *= math.sqrt(2 * math.pi)
        return params

    def _param_bounds(self) -> dict[str, tuple[float, float]]:
        return {'scale': (0.0, np.inf)}

    def fwhm(self, params: dict[str, sc.Variable]) -> sc.Variable:
        """Compute full width at half maximum.

        Parameters
        ----------
        params:
            Parameter values for which to compute the FWHM.

        Returns
        -------
        :
            FWHM of the model at the given parameters.
        """
        return 2 * math.sqrt(2 * math.log(2)) * params[self._prefix + 'scale']


class LorentzianModel(Model):
    r"""A Lorentzian function with arbitrary normalization.

    The model implements

    .. math::

        f(x; A, \mu, \sigma) = \frac{A}{\pi} \frac{\sigma}{{(x-\mu)}^2 + \sigma^2}

    with parameters

    - :math:`A`: ``'amplitude'``
    - :math:`\mu`: ``'loc'``
    - :math:`\sigma`: ``'scale'``
    """

    def __init__(self, *, prefix: str = '') -> None:
        """Initialize a Lorentzian model.

        Parameters
        ----------
        prefix:
            Prefix for model parameter names.
        """
        super().__init__(prefix=prefix, param_names=('amplitude', 'loc', 'scale'))

    def _call(self, x: sc.Variable, params: dict[str, sc.Variable]) -> sc.Variable:
        return _lorentzian(x, **params)

    def _guess(self, x: sc.Variable, y: sc.Variable) -> dict[str, sc.Variable]:
        params = _guess_from_peak(x, y)
        # Fudge factor taken from lmfit.
        # Not sure where exactly it comes from, but it is related to the normalization
        # of a Lorentzian and is approximately
        # 3.0 * math.pi / math.sqrt(2 * math.pi)
        params['amplitude'] *= 3.75
        return params

    def _param_bounds(self) -> dict[str, tuple[float, float]]:
        return {'scale': (0.0, np.inf)}

    def fwhm(self, params: dict[str, sc.Variable]) -> sc.Variable:
        """Compute full width at half maximum.

        Parameters
        ----------
        params:
            Parameter values for which to compute the FWHM.

        Returns
        -------
        :
            FWHM of the model at the given parameters.
        """
        return 2 * params[self._prefix + 'scale']


class PseudoVoigtModel(Model):
    r"""A Pseudo-Voigt function.

    The model implements

    .. math::

        f(x; A, \mu, \sigma, \alpha) = \alpha L(x; A, \mu, \sigma)
          + (1-\alpha) G(x; A, \mu, \sigma_G)

    where :math:`L` is a :class:`Lorentzian <LorentzianModel>`
    and :math:`G` is a :class:`Gaussian <GaussianModel>`.
    :math:`\sigma_G` is derived from :math:`\sigma` such that :math:`L` and
    :math:`G` have the same FWHM.

    It has parameters

    - :math:`A`: ``'amplitude'``
    - :math:`\mu`: ``'loc'``
    - :math:`\sigma`: ``'scale'``
    - :math:`\alpha`: ``'fraction'``
    """

    def __init__(self, *, prefix: str = '') -> None:
        super().__init__(
            prefix=prefix, param_names=('amplitude', 'loc', 'scale', 'fraction')
        )

    def _call(self, x: sc.Variable, params: dict[str, sc.Variable]) -> sc.Variable:
        params = dict(params.items())
        fraction = params.pop('fraction')

        lorentzian = _lorentzian(x, **params)
        # Adjust Gaussian scale such that Gaussian and Lorentzian have the same
        # FWHM of 2*scale.
        scale_g = params['scale'] / math.sqrt(2 * math.log(2))
        gaussian = _gaussian(
            x, amplitude=params['amplitude'], loc=params['loc'], scale=scale_g
        )

        return fraction * lorentzian + (1 - fraction) * gaussian

    def _guess(self, x: sc.Variable, y: sc.Variable) -> dict[str, sc.Variable]:
        params = _guess_from_peak(x, y)
        params['fraction'] = sc.scalar(0.5)
        # See Lorentzian
        params['amplitude'] *= 3.75
        return params

    def _param_bounds(self) -> dict[str, tuple[float, float]]:
        return {'scale': (0.0, np.inf), 'fraction': (0.0, 1.0)}

    def fwhm(self, params: dict[str, sc.Variable]) -> sc.Variable:
        """Compute full width at half maximum.

        Parameters
        ----------
        params:
            Parameter values for which to compute the FWHM.

        Returns
        -------
        :
            FWHM of the model at the given parameters.
        """
        # Note that the Gaussian component has an adjusted scale to enable this.
        return 2 * params[self._prefix + 'scale']


def _gaussian(
    x: sc.Variable, *, amplitude: sc.Variable, loc: sc.Variable, scale: sc.Variable
) -> sc.Variable:
    # Avoid division by 0
    scale = sc.scalar(max(scale.value, 1e-15), variance=scale.variance, unit=scale.unit)

    val = x - loc
    val *= val
    val /= -2 * scale**2
    val = sc.exp(val, out=val)
    val *= amplitude / (math.sqrt(2 * math.pi) * scale)
    return val


def _lorentzian(
    x: sc.Variable, *, amplitude: sc.Variable, loc: sc.Variable, scale: sc.Variable
) -> sc.Variable:
    # Avoid division by 0
    scale = sc.scalar(max(scale.value, 1e-15), variance=scale.variance, unit=scale.unit)

    val = x - loc
    val *= val
    val += scale**2
    val = sc.reciprocal(val, out=val)
    val *= amplitude * scale / math.pi
    return val


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
        # Exact gamma = FWHM / 2 for Lorentzian.
        scale = (x_half_max.max() - x_half_max.min()) / 2.0
    else:
        loc = x[np.argmax(y.values)]
        # 6.0 taken from lmfit, don't know where it comes from.
        scale = (sc.max(x) - sc.min(x)) / 6.0

    amplitude = scale * (y_max - y_min)
    return {
        'amplitude': amplitude,
        'loc': loc,
        'scale': scale,
    }
