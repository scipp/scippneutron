# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipp as sc
from scipp.scipy.optimize import curve_fit
from scipy.stats import chi2 as _scipy_chi2

from .model import (
    GaussianModel,
    LorentzianModel,
    Model,
    PolynomialModel,
    PseudoVoigtModel,
)


@dataclass
class FitResult:
    popt: dict[str, sc.Variable]
    red_chisq: sc.Variable
    p_value: sc.Variable
    aic: sc.Variable
    assessment: FitAssessment
    peak: Model
    background: Model
    window: sc.Variable
    message: str

    @classmethod
    def for_too_narrow_window(
        cls,
        peak: Model,
        background: Model,
        window: sc.Variable,
    ) -> FitResult:
        return cls.for_failure(
            assessment=FitAssessment.window_too_narrow,
            peak=peak,
            background=background,
            window=window,
        )

    @classmethod
    def for_failure(
        cls,
        *,
        assessment: FitAssessment | None = None,
        peak: Model,
        background: Model,
        window: sc.Variable,
        message: str | None = None,
    ) -> FitResult:
        return cls(
            popt={
                name: sc.scalar(np.nan)
                for name in peak.param_names | background.param_names
            },
            red_chisq=sc.scalar(-np.nan),
            p_value=sc.scalar(-np.nan),
            aic=sc.scalar(-np.inf),
            assessment=FitAssessment.failed if assessment is None else assessment,
            peak=peak,
            background=background,
            window=window,
            message=message or _message_from_assessment(assessment),
        )

    @property
    def success(self) -> bool:
        return self.assessment.success

    def better_than(self, other: FitResult) -> bool:
        return sc.all(self.aic < other.aic).value

    def eval_model(self, data: sc.Variable) -> sc.Variable:
        return (self.background + self.peak)(
            data, **{name: sc.values(val) for name, val in self.popt.items()}
        )

    def eval_peak(self, data: sc.Variable) -> sc.Variable:
        return self.peak(
            data,
            **{
                name: sc.values(val)
                for name, val in self.popt.items()
                if name.startswith('peak_')
            },
        )


class FitAssessment(enum.Enum):
    accept = enum.auto()
    candidate = enum.auto()
    failed = enum.auto()
    background_is_better = enum.auto()
    peak_too_narrow = enum.auto()
    peak_too_wide = enum.auto()
    peak_near_edge = enum.auto()
    peak_points_down = enum.auto()
    p_too_small = enum.auto()
    window_too_narrow = enum.auto()

    @property
    def success(self) -> bool:
        return self in (
            FitAssessment.accept,
            FitAssessment.candidate,
        )


def fit_peaks(
    data: sc.DataArray,
    *,
    peak_estimates: sc.Variable,
    windows: sc.Variable,
    background: Model | str | Iterable[Model] | Iterable[str],
    peak: Model | str | Iterable[Model] | Iterable[str],
) -> list[FitResult]:
    background = _parse_model_spec(background, prefix='bkg_')
    peak = _parse_model_spec(peak, prefix='peak_')

    if not sc.issorted(data.coords[data.dim], data.dim, order='ascending'):
        # A lot of code here assumes a sorted coord, either to use O(1) instead of O(n)
        # operations or to allow extracting windows.
        raise sc.CoordError(
            'fit_peaks requires the coordinate to be sorted in ascending order. '
            'Consider using scipp.sort to fix this.'
        )

    if windows.ndim == 0:
        windows = _fit_windows(data, peak_estimates, windows)

    results = []
    for i in range(windows.sizes[peak_estimates.dim]):
        window = windows[peak_estimates.dim, i]
        data_in_window = data[data.dim, window[0] : window[1]]
        results.append(_fit_peak_sc(data_in_window, window, background, peak))
    return results


def _fit_peak_sc(
    data: sc.DataArray,
    window: sc.Variable,
    backgrounds: tuple[Model, ...],
    peaks: tuple[Model, ...],
) -> FitResult:
    candidate_result = None
    # Loop order chosen as [(p0, b0), (p0, b1), (p1, b0), (p1, b1), ...]
    # because trying different background models usually improves fits more than
    # different peak models.
    for peak, background in itertools.product(peaks, backgrounds):
        result = _fit_peak_single_model(
            data, peak=peak, background=background, window=window
        )
        if candidate_result is None:
            candidate_result = result
        match result.assessment:
            case FitAssessment.candidate if result.better_than(candidate_result):
                candidate_result = result
            case FitAssessment.accept:
                return result
            # else: reject

    return candidate_result


def _fit_peak_single_model(
    data: sc.DataArray, peak: Model, background: Model, window: sc.Variable
) -> FitResult:
    model = background + peak
    bkg_p0 = _guess_background(data, model=background)
    p0 = {
        **bkg_p0,
        **_guess_peak(data, model=peak),
    }
    # TODO get from model
    bounds = {
        'peak_amplitude': (0.0, np.inf),
        'peak_scale': (0.0, np.inf),
        'peak_fraction': (0.0, 1.0),
    }

    if len(data) < len(p0):
        # not enough points to fit all parameters
        return FitResult.for_too_narrow_window(
            peak=peak, background=background, window=window
        )

    bkg_goodness_stats = _fit_background(background, data, bkg_p0)
    try:
        popt, goodness_stats = _perform_fit(model, data, p0=p0, bounds=bounds)
    except RuntimeError as err:
        return FitResult.for_failure(
            peak=peak,
            background=background,
            window=window,
            message=str(err.args[0]),
        )

    assessment = assess_fit(data, peak, popt, goodness_stats, bkg_goodness_stats)
    return FitResult(
        popt=popt,
        assessment=assessment,
        peak=peak,
        background=background,
        window=window,
        message=_message_from_assessment(assessment),
        **goodness_stats,
    )


def _fit_background(
    model: Model, data: sc.DataArray, p0: dict[str, sc.Variable]
) -> dict[str, sc.Variable] | None:
    try:
        _, goodness_stats = _perform_fit(model, data, p0, bounds={})
    except RuntimeError:
        # Background fits may fail when the background is a bad model.
        # Continue with a background+peak fit instead of aborting.
        return None
    return goodness_stats


def _perform_fit(
    model: Model,
    data: sc.DataArray,
    p0: dict[str, sc.Variable],
    bounds: dict[str, tuple[float, float]],
) -> tuple[dict[str, sc.Variable], dict[str, sc.Variable]]:
    # Workaround for https://github.com/scipp/scipp/issues/3418
    def fit_model(x: sc.Variable, **params: sc.Variable) -> sc.Variable:
        return model(x, **params)

    popt, _ = curve_fit(fit_model, data, p0=p0, bounds=bounds)
    best_fit = sc.DataArray(
        model(data.coords[data.dim], **{k: sc.values(p) for k, p in popt.items()}),
        coords=data.coords,
    )
    goodness_stats = _goodness_of_fit_statistics(data, best_fit, popt)
    return popt, goodness_stats


def _goodness_of_fit_statistics(
    data: sc.DataArray, best_fit: sc.DataArray, params: dict[str, sc.Variable]
) -> dict[str, sc.Variable]:
    # number of degrees of freedom
    n_dof = len(data) - len(params)

    chi_square = _chi_square(data, best_fit)
    reduced_chi_square = chi_square / n_dof
    p = sc.scalar(1 - _scipy_chi2(n_dof).cdf(chi_square.value))

    aic = _akaike_information_criterion(data, chi_square, params)

    return {
        'red_chisq': reduced_chi_square,
        'p_value': p,
        'aic': aic,
    }


def _chi_square(data: sc.DataArray, best_fit: sc.DataArray) -> sc.Variable:
    aux = (sc.values(data) - best_fit) ** 2
    aux /= sc.variances(data)
    return sc.sum(aux.data).to(unit='one')


def _akaike_information_criterion(
    data: sc.DataArray, chi_square: sc.Variable, params: dict
) -> sc.Variable:
    neg2_log_likelihood = len(data) * sc.log(chi_square / len(data))
    return neg2_log_likelihood + 2 * len(params)


def assess_fit(
    data: sc.DataArray,
    peak: Model,
    popt: dict[str, sc.Variable],
    goodness_stats: dict[str, sc.Variable],
    bkg_goodness_stats: dict[str, sc.Variable] | None,
) -> FitAssessment:
    if bkg_goodness_stats is not None:
        if bkg_goodness_stats['aic'] < goodness_stats['aic']:
            return FitAssessment.background_is_better
    if (goodness_stats['p_value'] < sc.scalar(0.01)).value:  # TODO tunable
        return FitAssessment.p_too_small
    if _peak_is_near_edge(data, popt):
        return FitAssessment.peak_near_edge
    if _curve_points_down(popt):
        return FitAssessment.peak_points_down
    if _peak_is_too_wide(data, peak, popt):
        return FitAssessment.peak_too_wide
    if _peak_is_too_narrow(data, peak, popt):
        return FitAssessment.peak_too_narrow
    return FitAssessment.accept


def _peak_is_near_edge(data: sc.DataArray, popt: dict[str, sc.Variable]) -> bool:
    coord = data.coords[data.dim]
    step_size = sc.min(coord[1:] - coord[:-1])
    return (popt['peak_loc'] - coord[0] < 2 * step_size).value or (
        coord[-1] - popt['peak_loc'] < 2 * step_size
    ).value


def _curve_points_down(popt: dict[str, sc.Variable]) -> bool:
    try:
        return popt['peak_amplitude'].value < 0
    except KeyError:
        return False


def _peak_is_too_wide(
    data: sc.DataArray, peak: Model, popt: dict[str, sc.Variable]
) -> bool:
    fwhm = peak.fwhm(popt)
    coord = data.coords[data.dim]
    return (fwhm > (coord[-1] - coord[0])).value  # TODO tunable


def _peak_is_too_narrow(
    data: sc.DataArray, peak: Model, popt: dict[str, sc.Variable]
) -> bool:
    fwhm = peak.fwhm(popt)
    coord = data.coords[data.dim]
    center_idx = np.argmin(abs(coord.values - popt['peak_loc'].values))
    # Average of bins around center index.
    # Bins don't normally vary quickly, so this is a good approximation.
    bin_width = (coord[center_idx + 1] - coord[center_idx - 1]) / 2
    return ((fwhm / bin_width) < sc.scalar(1.0)).value  # TODO tunable


def _guess_background(data: sc.DataArray, model: Model) -> dict[str, sc.Variable]:
    n = len(data) // 4  # TODO tunable
    tails = sc.concat([data[:n], data[-n:]], dim=data.dim)
    return model.guess(tails)


def _guess_peak(data: sc.DataArray, model: Model) -> dict[str, sc.Variable]:
    n = len(data) // 4  # TODO tunable (in sync with _guess_background?)
    bulk = data[n:-n]
    return model.guess(bulk)


def _fit_windows(
    data: sc.DataArray, center: sc.Variable, width: sc.Variable
) -> sc.Variable:
    windows = sc.empty(sizes={data.dim: len(center), 'range': 2}, unit=center.unit)
    windows['range', 0] = center - width / 2
    windows['range', 1] = np.nextafter(center.values + width.value / 2, np.inf)

    windows = _clip_to_data_range(data, windows)
    _separate_from_neighbors_in_place(center, windows)

    return windows


def _clip_to_data_range(data: sc.DataArray, windows: sc.Variable) -> sc.Variable:
    lo = data.coords[data.dim].min()
    hi = data.coords[data.dim].max()
    windows = sc.where(windows < lo, lo, windows)
    windows = sc.where(windows > hi, hi, windows)
    return windows


def _separate_from_neighbors_in_place(
    center: sc.Variable, windows: sc.Variable
) -> None:
    if not sc.issorted(center, center.dim):
        # Needed to easily identify neighbors.
        raise ValueError('Fit window centers must be sorted')

    left_neighbor = center[:-1]
    right_neighbor = center[1:]
    min_separation = (right_neighbor - left_neighbor) * (1 / 3)  # TODO tunable
    lo = left_neighbor + min_separation
    hi = right_neighbor - min_separation
    # Do not adjust the left edge of the first window and the right edge of the
    # last window because there are no neighbors on those sides.
    left_edge = windows['range', 0][1:]
    right_edge = windows['range', 1][:-1]
    left_edge[:] = sc.where(left_edge < lo, lo, left_edge)
    right_edge[:] = sc.where(right_edge > hi, hi, right_edge)


def _message_from_assessment(assessment: FitAssessment | None) -> str:
    match assessment:
        case FitAssessment.accept:
            return 'success'
        case FitAssessment.candidate:
            return 'success'
        case FitAssessment.peak_too_narrow:
            return 'peak too narrow'
        case FitAssessment.peak_too_wide:
            return 'peak too wide'
        case FitAssessment.peak_points_down:
            return 'wrong sign'
        case FitAssessment.window_too_narrow:
            return 'window too narrow'
        case FitAssessment.peak_near_edge:
            return 'too close to edge'
        case FitAssessment.p_too_small:
            return 'p-value too small'
        case FitAssessment.background_is_better:
            return 'background is better'
        case FitAssessment.failed:
            return 'failure'


def _parse_model_spec(
    spec: Model | str | Iterable[Model] | Iterable[str], prefix: str
) -> tuple[Model, ...]:
    if isinstance(spec, (Model, str)):
        spec = (spec,)
    if not spec:
        raise ValueError(f"No models specified for '{prefix}'")
    return tuple(_parse_single_model_spec(s, prefix=prefix) for s in spec)


def _parse_single_model_spec(spec: Model | str, prefix: str) -> Model:
    match spec:
        case Model():
            return spec.with_prefix(prefix)
        case 'linear':
            return PolynomialModel(degree=1, prefix=prefix)
        case 'quadratic':
            return PolynomialModel(degree=2, prefix=prefix)
        case 'gaussian':
            return GaussianModel(prefix=prefix)
        case 'lorentzian':
            return LorentzianModel(prefix=prefix)
        case 'pseudo_voigt':
            return PseudoVoigtModel(prefix=prefix)
        case _:
            raise ValueError(f"Unknown model: '{spec}'")
