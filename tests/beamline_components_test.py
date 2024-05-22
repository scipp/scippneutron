# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import numpy as np
import pytest
import scipp as sc

import scippneutron as scn


@pytest.fixture()
def data_array():
    positions = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return sc.DataArray(
        sc.Variable(dims=['position', 'tof'], values=np.random.rand(3, 9)),
        coords={
            'position': sc.vectors(dims=['position'], values=positions, unit='m'),
            'source_position': sc.vector(value=np.array([0, 0, -10]), unit='m'),
            'sample_position': sc.vector(value=np.array([0, 1, 0]), unit='m'),
        },
    )


def test_position(data_array):
    assert sc.identical(scn.position(data_array), data_array.coords['position'])


def test_source_position(data_array):
    assert sc.identical(
        scn.source_position(data_array),
        sc.vector(value=np.array([0, 0, -10]), unit='m'),
    )


def test_sample_position(data_array):
    assert sc.identical(
        scn.sample_position(data_array), sc.vector(value=np.array([0, 1, 0]), unit='m')
    )


def test_L1(data_array):
    assert sc.identical(scn.L1(data_array), sc.scalar(np.sqrt(1 + 100), unit='m'))


def test_L2(data_array):
    assert sc.identical(
        scn.L2(data_array),
        sc.array(dims=['position'], values=np.sqrt([1 + 1, 0, 1 + 1]), unit='m'),
    )


def test_Ltotal_scatter(data_array):
    assert sc.identical(
        scn.Ltotal(data_array, scatter=True), scn.L1(data_array) + scn.L2(data_array)
    )


def test_Ltotal_no_scatter(data_array):
    assert sc.identical(
        scn.Ltotal(data_array, scatter=False),
        sc.array(dims=['position'], values=np.sqrt([1 + 100, 1 + 100, 121]), unit='m'),
    )


def test_two_theta(data_array):
    # Bare-bones test because computation of 2theta is complicated,
    # and it is tested separately elsewhere.
    two_theta = scn.two_theta(data_array)
    assert two_theta.unit == 'rad'
    assert two_theta.sizes == scn.position(data_array).sizes
