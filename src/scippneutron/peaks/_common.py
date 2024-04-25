# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class FitParameters:
    """Parameters for fitting peaks."""

    guess_background_fraction: float = 0.5
    """The fraction of fit windows used for estimating the background.

    Determines what fraction of each fit window is used for estimating the initial
    parameters for background fits.
    E.g., a value of 0.5 means that half the range is used for the background;
    specifically first and last quarter of the window.
    Initial parameters for the peak are estimated from the remainder of the window.
    """
    neighbor_separation_factor: float = 1 / 3
    r"""Determines how close fit windows may be to neighboring peaks.

    The fit window for a peak with initial estimate :math:`p_i` is shrunk to the
    exclusive interval :math:`(p_i - \Delta, p_i + \Delta)` if it is smaller
    than the given fit window, where

    .. math::

        \Delta = (p_{i+1} - p_{i-1}) \cdot \mathsf{neighbor\_separation\_factor}

    is based on the separation between the left and right neighbor estimate of the peak.

    Importantly, this uses the peak *estimates*, not the actual fitted peak positions.
    """


@dataclass(kw_only=True, slots=True)
class FitRequirements:
    """Requirements that fitted models must satisfy to be considered successful."""

    min_p_value: float = 0.01
    """Minimum for the p-value of the fit.

    See :attr:`FitResult.p_value`.
    """
    max_peak_width_factor: float = 1.0
    r"""Maximum allowed width of peaks relative to fit window.

    The full width at half maximum of fitted peaks must satisfy

    .. math::

        \mathsf{FWHM} < \mathsf{max\_peak\_width\_factor} \cdot \mathsf{window\_width}
    """
    min_peak_width_factor: float = 1.0
    r"""Minimum allowed width of peaks relative to coordinate spacing.

    The full width at half maximum of fitted peaks must satisfy

    .. math::

        \mathsf{FWHM} < \mathsf{min\_peak\_width\_factor} \cdot \Delta_{\mathsf{coord}}

    where :math:`\Delta_{\mathsf{coord}}` is the spacing of the coordinate around
    the peak center.
    """
