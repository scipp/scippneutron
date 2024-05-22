# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import pytest
import scipp as sc
from scipp.testing import assert_identical

from scippneutron.tof import fakes


@pytest.fixture()
def ess_10s_14Hz() -> fakes.FakeSource:
    return fakes.FakeSource(
        frequency=sc.scalar(14.0, unit='Hz'), run_length=sc.scalar(10.0, unit='s')
    )


def test_fake_source(ess_10s_14Hz) -> None:
    source = ess_10s_14Hz
    assert source.t0.unit == sc.Unit('ns')
    assert source.t0.sizes == {'pulse': 140}
    assert_identical(source.t0[0], sc.datetime('2019-12-25T06:00:00.0', unit='ns'))
    assert_identical(
        source.t0[-1],
        sc.datetime('2019-12-25T06:00:00.0', unit='ns')
        + sc.scalar(139 / 140 * 10 * 1e9, unit='ns', dtype='int64'),
    )


def test_fake_monitor(ess_10s_14Hz) -> None:
    pulse = fakes.FakePulse(
        time_min=sc.scalar(0.0, unit='ms'),
        time_max=sc.scalar(3.0, unit='ms'),
        wavelength_min=sc.scalar(0.1, unit='angstrom'),
        wavelength_max=sc.scalar(10.0, unit='angstrom'),
    )
    beamline = fakes.FakeBeamline(
        source=ess_10s_14Hz,
        pulse=pulse,
        choppers=fakes.wfm_choppers,
        monitors={'monitor': sc.scalar(26.0, unit='m')},
        detectors={},
    )
    mon, _ = beamline.get_monitor('monitor')
    assert mon.sizes == {'pulse': 140}
