# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass

import lmfit
import numpy as np
import scipp as sc


@dataclass
class FitResult:
    lm_result: lmfit.model.ModelResult
    best_fit: sc.DataArray
    assessment: FitAssessment

    @property
    def success(self) -> bool:
        return self.assessment in (FitAssessment.accept, FitAssessment.candidate)

    def better_than(self, other: FitResult) -> bool:
        return self.lm_result.aic > other.lm_result.aic


class FitAssessment(enum.Enum):
    accept = enum.auto()
    candidate = enum.auto()
    reject = enum.auto()
    peak_too_narrow = enum.auto()
    peak_too_wide = enum.auto()
    peak_points_down = enum.auto()
    chisq_too_large = enum.auto()
    window_too_narrow = enum.auto()


def fit_peaks(
    data: sc.DataArray, peak_estimates: sc.Variable, windows: sc.Variable
) -> list[FitResult]:
    if windows.ndim == 0:
        windows = _fit_windows(data, peak_estimates, windows)

    results = []
    for estimate, window in zip(
        peak_estimates, windows.transpose((peak_estimates.dim, 'range')).values
    ):
        data_in_window = data[
            data.dim, window[0] * windows.unit : window[1] * windows.unit
        ]
        results.append(_fit_peak_sc(data_in_window, estimate))
    return results


def _fit_peak(data: sc.DataArray, peak_estimate: sc.Variable) -> FitResult:
    background_models = (
        # lmfit.models.ExponentialModel(prefix='bkg_'),
        lmfit.models.LinearModel(prefix='bkg_'),
        # lmfit.models.QuadraticModel(prefix='bkg_'),
    )
    peak_models = (
        lmfit.models.GaussianModel(prefix='peak_'),
        # lmfit.models.LorentzianModel(prefix='peak_'),
        # lmfit.models.PseudoVoigtModel(prefix='peak_'),
        # lmfit.models.VoigtModel(prefix='peak_'),
    )

    x = data.coords[data.dim].values
    y = data.values
    e = data.variances

    candidate_result = None
    for peak, background in itertools.product(peak_models, background_models):
        params = _guess_background(x, y, background)
        params.update(_guess_peak(x, y, peak))
        model = background + peak

        if len(x) < len(params):
            if candidate_result is None:
                candidate_result = FitResult(
                    lm_result=None,
                    best_fit=None,
                    assessment=FitAssessment.window_too_narrow,
                )
            continue  # not enough points to fit all parameters

        # TODO weights correct?
        lm_result = model.fit(y, x=x, params=params, weights=1 / np.sqrt(e))
        best_fit = sc.DataArray(
            sc.array(dims=[data.dim], values=lm_result.best_fit, unit=data.unit),
            coords=data.coords,
        )
        assessment = assess_fit(data, lm_result)
        result = FitResult(
            lm_result=lm_result, best_fit=best_fit, assessment=assessment
        )

        if assessment == FitAssessment.candidate:
            if candidate_result is None:
                candidate_result = result
            else:
                if result.better_than(candidate_result):
                    candidate_result = result
        elif assessment == FitAssessment.accept:
            return result
        else:
            if candidate_result is None:
                candidate_result = result

    return candidate_result


def _fit_peak_sc(data: sc.DataArray, peak_estimate: sc.Variable) -> FitResult:
    from math import pi, sqrt

    from scipp.scipy.optimize import curve_fit

    s2pi = sqrt(2 * pi)
    # s2 = sqrt(2.0)
    # # tiny had been numpy.finfo(numpy.float64).eps ~=2.2e16.
    # # here, we explicitly set it to 1.e-15 == numpy.finfo(numpy.float64).resolution
    # tiny = 1.0e-15

    def model(x, *, bkg_slope, bkg_intercept, peak_amplitude, peak_center, peak_sigma):
        # TODO lmfit uses max(tiny, ...) to prevent div by 0
        bkg = bkg_slope * x + bkg_intercept
        peak = (
            peak_amplitude
            / (s2pi * peak_sigma)
            * sc.exp(-((x - peak_center) ** 2) / (2 * peak_sigma**2))
        )
        return bkg + peak

    # Taken from lmfit
    def guess_from_peak(y, x):
        """Estimate starting values from 1D peak data and create Parameters."""
        sort_increasing = np.argsort(x)
        x = x[sort_increasing]
        y = y[sort_increasing]

        maxy, miny = max(y), min(y)
        maxx, minx = max(x), min(x)
        cen = x[np.argmax(y)]
        height = (maxy - miny) * 3.0
        sig = (maxx - minx) / 6.0

        # the explicit conversion to a NumPy array is to make sure that the
        # indexing on line 65 also works if the data is supplied as pandas.Series
        x_halfmax = np.array(x[y > (maxy + miny) / 2.0])
        if len(x_halfmax) > 2:
            sig = (x_halfmax[-1] - x_halfmax[0]) / 2.0
            cen = x_halfmax.mean()
        amp = height * sig  # because of 1/sigma in gaussian?
        sig = sig
        return amp, cen, sig

    x = data.coords[data.dim].values
    y = data.values
    n = len(x) // 4
    x = np.r_[x[:n], x[-n:]]
    y = np.r_[y[:n], y[-n:]]
    slope, intercept = np.polyfit(x, y, 1)

    x = data.coords[data.dim].values
    y = data.values
    n = len(x) // 4
    x = x[n:-n]
    y = y[n:-n]
    amp, cen, sig = guess_from_peak(y, x)

    params = {
        'bkg_slope': sc.scalar(slope, unit=data.unit / sc.Unit('Å')),
        'bkg_intercept': sc.scalar(intercept, unit=data.unit),
        'peak_amplitude': sc.scalar(amp, unit=data.unit * sc.Unit('Å')),
        'peak_center': sc.scalar(cen, unit='Å'),
        'peak_sigma': sc.scalar(sig, unit='Å'),
    }

    try:
        popt, _ = curve_fit(model, data, p0=params)
    except RuntimeError:
        return FitResult(lm_result=None, best_fit=data, assessment=FitAssessment.reject)
    best_fit = sc.DataArray(
        model(data.coords[data.dim], **{k: sc.values(p) for k, p in popt.items()}),
        coords=data.coords,
    )
    return FitResult(lm_result=None, best_fit=best_fit, assessment=FitAssessment.accept)


def assess_fit(data: sc.DataArray, result: lmfit.model.ModelResult) -> FitAssessment:
    if not result.success:
        return FitAssessment.reject
    if result.redchi > 100:  # TODO tunable
        # TODO mantid checks for chisq < 0
        # https://github.com/mantidproject/mantid/blob/f03bd8cd7087aeecc5c74673af93871137dfb13a/Framework/Algorithms/src/StripPeaks.cpp#L203
        return FitAssessment.chisq_too_large
    if _curve_points_down(result):
        return FitAssessment.peak_points_down
    if _peak_is_too_wide(data, result):
        return FitAssessment.peak_too_wide
    if _peak_is_too_narrow(data, result):
        return FitAssessment.peak_too_narrow
    return FitAssessment.accept


def _curve_points_down(result: lmfit.model.ModelResult) -> bool:
    try:
        return result.params['peak_amplitude'].value < 0
    except KeyError:
        return False


def _peak_is_too_wide(data: sc.DataArray, result: lmfit.model.ModelResult) -> bool:
    fwhm = result.params['peak_fwhm'].value
    coord = data.coords[data.dim].values
    return fwhm > (coord[-1] - coord[0])  # TODO tunable


def _peak_is_too_narrow(data: sc.DataArray, result: lmfit.model.ModelResult) -> bool:
    fwhm = result.params['peak_fwhm'].value
    coord = data.coords[data.dim].values
    center_idx = np.argmin(np.abs(coord - result.params['peak_center'].value))
    # Average of bins around center index.
    # Bins don't normally vary quickly, so this is a good approximation.
    bin_width = (coord[center_idx + 1] - coord[center_idx - 1]) / 2
    return (fwhm / bin_width) < 1.5  # TODO tunable


def _guess_background(x, y, model):
    n = len(x) // 4  # TODO tunable
    x = np.r_[x[:n], x[-n:]]
    y = np.r_[y[:n], y[-n:]]
    return model.guess(y, x=x)


def _guess_peak(x, y, model):
    n = len(x) // 4  # TODO tunable (in sync with _guess_background?)
    x = x[n:-n]
    y = y[n:-n]
    return model.guess(y, x=x)


def _fit_windows(
    data: sc.DataArray, center: sc.Variable, width: sc.Variable
) -> sc.Variable:
    windows = sc.empty(sizes={data.dim: len(center), 'range': 2}, unit=center.unit)
    windows['range', 0] = center - width / 2
    windows['range', 1] = center + width / 2

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
