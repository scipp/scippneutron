# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @file
# @author Simon Heybrock
import scipp as sc
import scippneutron as scn
import numpy as np
from .common import make_dataset_with_beamline


def test_neutron_beamline():
    d = make_dataset_with_beamline()

    assert sc.identical(scn.source_position(d),
                        sc.vector(value=np.array([0, 0, -10]), unit=sc.units.m))
    assert sc.identical(scn.sample_position(d),
                        sc.vector(value=np.array([0, 0, 0]), unit=sc.units.m))
    assert sc.identical(scn.L1(d), 10.0 * sc.units.m)
    assert sc.identical(
        scn.L2(d), sc.Variable(dims=['position'], values=np.ones(4), unit=sc.units.m))
    two_theta = scn.two_theta(d)
    assert sc.identical(scn.L1(d) + scn.L2(d), scn.Ltotal(d, scatter=True))
    assert two_theta.unit == sc.units.rad
    assert two_theta.dims == ('position',)
