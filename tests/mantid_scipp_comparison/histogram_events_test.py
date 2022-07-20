# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

import pytest
import scipp as sc
import scippneutron as scn

try:
    import mantid.simpleapi as sapi
    import mantid.kernel as kernel
except ImportError:
    pytestmark = pytest.mark.skip('Mantid framework is unavailable')
    sapi = None
    kernel = None


@pytest.fixture
def in_ws():
    ws = sapi.Load(Filename=scn.data.get_path("CNCS_51936_event.nxs"), StoreInADS=False)
    return ws


@pytest.fixture
def in_da(in_ws):
    return scn.mantid.from_mantid(in_ws).astype('float64')


def test_histogram_events(in_ws, in_da):
    out_ws = sapi.Rebin(InputWorkspace=in_ws,
                        Params=[0, 10, 1000],
                        PreserveEvents=False,
                        StoreInADS=False)
    out_mantid = scn.mantid.from_mantid(out_ws)
    out_scipp = in_da.hist(
        tof=sc.linspace('tof', 0, 1000, num=101, dtype='float64', unit='us'))
    assert sc.utils.comparison.isnear(out_scipp, out_mantid, rtol=1e-15 * sc.units.one)
