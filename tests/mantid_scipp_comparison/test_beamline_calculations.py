# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

import pytest
import scippneutron as scn
import scipp as sc

try:
    import mantid.simpleapi as sapi
    import mantid.kernel as kernel
except ImportError:
    pytestmark = pytest.mark.skip('Mantid framework is unavailable')
    sapi = None
    kernel = None


@pytest.fixture
def in_ws():
    ws = sapi.CreateSampleWorkspace(SourceDistanceFromSample=10.0,
                                    BankDistanceFromSample=1.1,
                                    BankPixelWidth=2,
                                    NumBanks=1,
                                    XMax=200,
                                    StoreInADS=False)
    return ws


@pytest.fixture
def in_da(in_ws):
    return scn.mantid.from_mantid(in_ws)


def test_beamline_compute_l1(in_ws, in_da):
    out_mantid = in_ws.detectorInfo().l1() * sc.Unit('m')
    in_da = scn.mantid.from_mantid(in_ws)
    out_scipp = scn.L1(in_da)
    assert sc.allclose(out_scipp,
                       out_mantid,
                       rtol=1e-15 * sc.units.one,
                       atol=1e-15 * out_scipp.unit)


def test_beamline_compute_l2(in_ws, in_da):
    out_mantid = sc.array(
        dims=['spectrum'],
        unit='m',
        values=[in_ws.detectorInfo().l2(i) for i in range(in_ws.detectorInfo().size())])
    in_da = scn.mantid.from_mantid(in_ws)
    out_scipp = scn.L2(in_da)
    assert sc.allclose(out_scipp,
                       out_mantid,
                       rtol=1e-15 * sc.units.one,
                       atol=1e-15 * out_scipp.unit)


def test_beamline_compute_two_theta(in_ws, in_da):
    out_mantid = sc.array(dims=['spectrum'],
                          unit='rad',
                          values=[
                              in_ws.detectorInfo().twoTheta(i)
                              for i in range(in_ws.detectorInfo().size())
                          ])
    in_da = scn.mantid.from_mantid(in_ws)
    out_scipp = scn.two_theta(in_da)
    assert sc.allclose(out_scipp,
                       out_mantid,
                       rtol=1e-13 * sc.units.one,
                       atol=1e-13 * out_scipp.unit)
