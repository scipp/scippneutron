# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc
import scipp.testing

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

    ub = alignment.ub_from_3_peaks(peaks)
    assert ub.sizes == {}
    assert ub.dtype == sc.DType.linear_transform3
    assert ub.unit == 'one'


def test_u_from_b_and_ub() -> None:
    quat = np.array([5, 2, 6, 3])
    original_u = sc.spatial.rotation(value=quat / np.linalg.norm(quat))
    b = sc.spatial.linear_transform(
        value=[[7, 2, 3], [7, 1, 2], [-5, -3, 9]], unit='one'
    )
    ub = original_u * b

    reconstructed_u = alignment.u_from_b_and_ub(b=b, ub=ub)

    original_u_matrix = original_u * sc.spatial.linear_transform(value=np.eye(3))
    sc.testing.assert_allclose(reconstructed_u, original_u_matrix)
