# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @file
# @author Simon Heybrock
import scipp as sc
import scippneutron as scn
import numpy as np


def make_dataset_with_beamline():
    positions = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]]
    d = sc.Dataset(
        data={'a': sc.Variable(dims=['position', 'tof'], values=np.random.rand(4, 9))},
        coords={
            'tof':
            sc.Variable(dims=['tof'],
                        values=np.arange(1000.0, 1010.0),
                        unit=sc.units.us),
            'position':
            sc.vectors(dims=['position'], values=positions, unit=sc.units.m)
        })

    d.coords['source_position'] = sc.vector(value=np.array([0, 0, -10]),
                                            unit=sc.units.m)
    d.coords['sample_position'] = sc.vector(value=np.array([0, 0, 0]), unit=sc.units.m)
    return d


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
    assert two_theta.dims == ['position']


def test_neutron_instrument_view_3d():
    d = make_dataset_with_beamline()
    scn.instrument_view(d["a"])


def test_neutron_instrument_view_with_dataset():
    d = make_dataset_with_beamline()
    d['b'] = sc.Variable(dims=['position', 'tof'], values=np.arange(36.).reshape(4, 9))
    scn.instrument_view(d)


def test_neutron_instrument_view_with_masks():
    d = make_dataset_with_beamline()
    x = np.transpose(d.coords['position'].values)[0, :]
    d['a'].masks['amask'] = sc.Variable(dims=['position'],
                                        values=np.less(np.abs(x), 0.5))
    scn.instrument_view(d["a"])


def test_neutron_instrument_view_with_cmap_args():
    d = make_dataset_with_beamline()
    scn.instrument_view(d["a"],
                        vmin=0.001 * sc.units.one,
                        vmax=5.0 * sc.units.one,
                        cmap="magma",
                        norm="log")
