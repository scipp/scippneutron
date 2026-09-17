# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import scipp as sc

from scippneutron.crystal import alignment


def test_ub_matrix_from_3_peaks() -> None:
    peaks = alignment.BraggPeaks(
        hkl=sc.vectors(
            dims=['p'], values=[[1, 0, 0], [0, 1, 0], [0, 1, 1]], unit='1/angstrom'
        ),
        q=sc.vectors(
            dims=['p'], values=[[1, 2, 3], [4, 5, 6], [-1, -2, -3]], unit='1/angstrom'
        ),
        r=sc.spatial.rotations_from_rotvecs(
            sc.vectors(
                dims=['p'],
                values=[[0, 0.1, 0], [0, 1.1, 0], [0, 0.8, 0]],
                unit='rad',
            )
        ),
    )

    ub = alignment.ub_matrix_from_3_peaks(peaks)
    assert ub.sizes == {}
    assert ub.dtype == sc.DType.linear_transform3
    assert ub.unit == 'one'
