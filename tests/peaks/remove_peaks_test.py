# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc
import scipp.testing

from scippneutron.peaks import FitAssessment, FitResult, remove_peaks
from scippneutron.peaks.model import GaussianModel, PolynomialModel


def test_remove_peaks_single_peak_clean():
    background = PolynomialModel(degree=1, prefix='bkg_')
    peak = GaussianModel(prefix='peak_')

    a0 = sc.scalar(140.0, unit='counts')
    a1 = sc.scalar(-6.1, unit='counts/Å')
    amplitude = sc.scalar(0.6, unit='counts*Å')
    loc = sc.scalar(1.5, unit='Å')
    scale = sc.scalar(0.04, unit='Å')
    params = {
        'bkg_a0': a0,
        'bkg_a1': a1,
        'peak_amplitude': amplitude,
        'peak_loc': loc,
        'peak_scale': scale,
    }

    x = sc.linspace('d', 0.2, 2.4, 200, unit='Å')
    y_bkg = background(x, bkg_a0=a0, bkg_a1=a1)
    y = y_bkg + peak(x, peak_amplitude=amplitude, peak_loc=loc, peak_scale=scale)
    with_peak = sc.DataArray(y, coords={'d': x})
    original = with_peak.copy()
    expected = sc.DataArray(y_bkg, coords={'d': x})

    result = FitResult(
        aic=sc.scalar(np.nan),
        assessment=FitAssessment.success,
        background=background,
        message='',
        p_value=sc.scalar(np.nan),
        peak=peak,
        popt=params,
        red_chisq=sc.scalar(np.nan),
        window=sc.array(dims=['range'], values=[1.2, 1.7], unit='Å'),
    )
    without_peak = remove_peaks(with_peak, [result])

    sc.testing.assert_identical(with_peak, original)
    sc.testing.assert_allclose(without_peak, expected, rtol=sc.scalar(1e-6))


def test_remove_peaks_single_peak_noisy():
    rng = np.random.default_rng(8381)

    background = PolynomialModel(degree=1, prefix='bkg_')
    peak = GaussianModel(prefix='peak_')

    a0 = sc.scalar(140.0, unit='counts')
    a1 = sc.scalar(-6.1, unit='counts/Å')
    amplitude = sc.scalar(0.6, unit='counts*Å')
    loc = sc.scalar(1.5, unit='Å')
    scale = sc.scalar(0.04, unit='Å')
    params = {
        'bkg_a0': a0,
        'bkg_a1': a1,
        'peak_amplitude': amplitude,
        'peak_loc': loc,
        'peak_scale': scale,
    }

    x = sc.linspace('d', 0.2, 2.4, 200, unit='Å')
    y_bkg = background(x, bkg_a0=a0, bkg_a1=a1) + sc.array(
        dims=x.dims, values=rng.normal(0.0, 0.3, x.shape), unit='counts'
    )
    y = y_bkg + peak(x, peak_amplitude=amplitude, peak_loc=loc, peak_scale=scale)
    with_peak = sc.DataArray(y, coords={'d': x})
    original = with_peak.copy()
    expected = sc.DataArray(y_bkg, coords={'d': x})

    result = FitResult(
        aic=sc.scalar(np.nan),
        assessment=FitAssessment.success,
        background=background,
        message='',
        p_value=sc.scalar(np.nan),
        peak=peak,
        popt=params,
        red_chisq=sc.scalar(np.nan),
        window=sc.array(dims=['range'], values=[1.2, 1.7], unit='Å'),
    )
    without_peak = remove_peaks(with_peak, [result])

    sc.testing.assert_identical(with_peak, original)
    sc.testing.assert_allclose(without_peak, expected, rtol=sc.scalar(1e-6))


def test_remove_peaks_two_peak_noisy():
    rng = np.random.default_rng(8381)

    background = PolynomialModel(degree=1, prefix='bkg_')
    peak = GaussianModel(prefix='peak_')
    a0 = sc.scalar(110.0, unit='counts')
    a1 = sc.scalar(-5.7, unit='counts/Å')
    amplitude1 = sc.scalar(0.6, unit='counts*Å')
    loc1 = sc.scalar(1.8, unit='Å')
    scale1 = sc.scalar(0.04, unit='Å')
    amplitude2 = sc.scalar(0.3, unit='counts*Å')
    loc2 = sc.scalar(0.9, unit='Å')
    scale2 = sc.scalar(0.05, unit='Å')
    params1 = {
        'bkg_a0': a0,
        'bkg_a1': a1,
        'peak_amplitude': amplitude1,
        'peak_loc': loc1,
        'peak_scale': scale1,
    }
    params2 = {
        'bkg_a0': a0,
        'bkg_a1': a1,
        'peak_amplitude': amplitude2,
        'peak_loc': loc2,
        'peak_scale': scale2,
    }

    x = sc.linspace('d', 0.2, 2.4, 200, unit='Å')
    y_bkg = background(x, bkg_a0=a0, bkg_a1=a1) + sc.array(
        dims=x.dims, values=rng.normal(0.0, 0.3, x.shape), unit='counts'
    )
    y = (
        y_bkg
        + peak(x, peak_amplitude=amplitude1, peak_loc=loc1, peak_scale=scale1)
        + peak(x, peak_amplitude=amplitude2, peak_loc=loc2, peak_scale=scale2)
    )
    with_peak = sc.DataArray(y, coords={'d': x})
    original = with_peak.copy()
    expected = sc.DataArray(y_bkg, coords={'d': x})

    result1 = FitResult(
        aic=sc.scalar(np.nan),
        assessment=FitAssessment.success,
        background=background,
        message='',
        p_value=sc.scalar(np.nan),
        peak=peak,
        popt=params1,
        red_chisq=sc.scalar(np.nan),
        window=sc.array(dims=['range'], values=[1.6, 2.2], unit='Å'),
    )
    result2 = FitResult(
        aic=sc.scalar(np.nan),
        assessment=FitAssessment.success,
        background=background,
        message='',
        p_value=sc.scalar(np.nan),
        peak=peak,
        popt=params2,
        red_chisq=sc.scalar(np.nan),
        window=sc.array(dims=['range'], values=[0.7, 1.1], unit='Å'),
    )
    without_peak = remove_peaks(with_peak, [result1, result2])

    sc.testing.assert_identical(with_peak, original)
    sc.testing.assert_allclose(without_peak, expected, rtol=sc.scalar(1e-5))
