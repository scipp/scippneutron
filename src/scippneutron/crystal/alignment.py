# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Alignment of single-crystal samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipp as sc

from ._linalg import invert_transform


@dataclass(frozen=True, slots=True)
class BraggPeaks:
    """Lattice coordinates and instrumental parameters for one or more Bragg peaks.

    All fields must have matching shapes.
    ``hkl`` and ``q`` must have dtype 'vector3' and ``r`` must have dtype 'rotation3'.
    """

    hkl: sc.Variable
    """The Miller indices of the peak."""
    q: sc.Variable
    """The lab-frame momentum transfer where the peak is observed."""
    r: sc.Variable
    """The sample rotation matrix corresponding the the observed momentum transfer."""

    def __post_init__(self) -> None:
        if self.r.dtype != sc.DType.rotation3:
            raise sc.DTypeError(
                f"Expected dtype 'rotation3' for 'r' but got {self.r.dtype}."
            )

        sizes = self.r.sizes

        for key in ('hkl', 'q'):
            item = getattr(self, key)
            if item.sizes != sizes:
                raise sc.DimensionError(
                    f"Mismatch between sizes: '{key}' has sizes {item.sizes} "
                    f"but expected {sizes}."
                )
            if item.dtype != sc.DType.vector3:
                raise sc.DTypeError(
                    f"Expected dtype 'vector3' for '{key}' but got {item.dtype}."
                )

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the peak coordinates."""
        return self.hkl.shape


def ub_matrix_from_3_peaks(peaks: BraggPeaks) -> sc.Variable:
    if peaks.shape != (3,):
        raise ValueError(f"Expected exactly 3 peaks, got {peaks.shape}.")

    try:
        r_inv_times_q = [
            invert_transform(r) * q for q, r in zip(peaks.q, peaks.r, strict=True)
        ]
    except ValueError as error:
        error.add_note("When inverting the sample rotation matrix.")
        raise
    q_mat = sc.spatial.linear_transform(
        value=np.array([x.value for x in r_inv_times_q]).T / (2 * np.pi),
        unit=peaks.q.unit / peaks.r.unit,
    )

    v_mat = sc.spatial.linear_transform(
        value=np.array([hkl.value for hkl in peaks.hkl]).T, unit=peaks.hkl.unit
    )
    try:
        v_mat_inv = invert_transform(v_mat)
    except ValueError as error:
        error.add_note("When inverting the V matrix (combination of hkl vectors).")
        raise

    return sc.to_unit(q_mat * v_mat_inv, "one")
