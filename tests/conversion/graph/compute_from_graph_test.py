# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

from collections.abc import Generator

import pytest
import scipp as sc

from scippneutron.conversion.graph import beamline, tof


def _flatten_keys(graph: dict[str, object]) -> Generator[str | tuple[str], None, None]:
    for key in graph:
        if isinstance(key, str):
            yield key
        else:
            yield from key


@pytest.mark.parametrize("target", _flatten_keys(tof.elastic('tof')))
def test_elastic_can_compute_all_from_tof(target: str | tuple[str]) -> None:
    graph = {**tof.elastic('tof'), **beamline.beamline(scatter=True)}
    da = sc.DataArray(
        sc.zeros(sizes={'tof': 4, 'detector': 2}),
        coords={
            'tof': sc.arange('tof', 5, unit='us'),
            'pulse_time': 0.1 * sc.arange('tof', 5, unit='us'),
            'position': sc.vectors(
                dims=['detector'], values=[[1, 2, 3], [4, 2, 1]], unit='m'
            ),
            'sample_position': sc.vector(value=[0, 0, 0], unit='m'),
            'source_position': sc.vector(value=[0, 0, -10], unit='m'),
            'sample_rotation': sc.spatial.rotation(value=[0, 0, 1, 1]),
            'b_matrix': sc.spatial.linear_transform(
                value=[[1, 0, 0], [0, 1, 0], [0, 1, 1]]
            ),
            'u_matrix': sc.spatial.linear_transform(
                value=[[0, 0.5, 0], [0.2, 1, 0], [0.1, 0, 1]]
            ),
        },
    )
    result = da.transform_coords(target, graph)
    assert target in result.coords
