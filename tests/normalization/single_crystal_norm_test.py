# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import numpy as np

from scippneutron.normalization import compute_q_de_norm


def test_single_crystal_norm_ins() -> None:
    h_min, h_max = 0.0, 0.0
    k_min, k_max = 0.0, 0.0
    l_min, l_max = 0.2, 0.9
    mom_min, mom_max = 1.0, 1.6
    trajectory_start = [h_min, k_min, l_min, mom_min]
    trajectory_stop = [h_max, k_max, l_max, mom_max]

    h_edges = np.linspace(-0.5, 0.5, 4)
    k_edges = np.linspace(-0.5, 0.5, 4)
    l_edges = np.linspace(0.1, 1.3, 5)
    mom_edges = np.linspace(0.5, 1.7, 4)
    edges = [h_edges, k_edges, l_edges, mom_edges]

    norm = compute_q_de_norm(
        trajectory_start=trajectory_start,
        trajectory_stop=trajectory_stop,
        all_edges=edges,
    )

    expected = np.zeros([3, 3, 4, 3])
    expected[1, 1] = [
        [0, 0.17142857142857149, 0],
        [0, 0.12857142857142856, 0.12857142857142856],
        [0, 0, 0.17142857142857149],
        [0, 0, 0],
    ]

    np.testing.assert_allclose(norm, expected)
