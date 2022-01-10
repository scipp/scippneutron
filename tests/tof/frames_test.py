# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import pytest
import numpy as np
import scipp as sc
from scippneutron.tof import frames


def array(*, npixel = 3, nevent = 1000):
    time_offset = sc.array(dims=['event'], values=np.random.rand(nevent)*71000, unit='us')
    pixel = sc.arange(dim='event', start=0, stop=nevent) % npixel
    events = sc.DataArray(sc.ones(sizes=time_offset.sizes), coords={'time_offset':time_offset, 'pixel':pixel})
    pixel = sc.arange(dim='pixel', start=0, stop=npixel)
    da = sc.bin(events, groups=[pixel])
    #da = sc.DataArray(sc.ones(dims=['pixel'], shape=[npixel]))

    da.coords['L1'] = sc.scalar(value=160.0, unit='m')
    da.coords['L2'] = sc.array(dims=['pixel'], values=[1.0,1.1,1.2], unit='m')
    return da


def test_make_frames():
    da = array()
    da = frames.make_frames(da,
                       frame_length=71.0 * sc.Unit('ms'),
                       frame_offset=30.1 * sc.Unit('ms'),
                       lambda_min=2.5 * sc.Unit('Angstrom'),
                       lambda_max=3.5 * sc.Unit('Angstrom'))
    print(da)
    assert da.sum().value == 1000


class TestMakeFrames:
    def test_ab(self):
        pass


#def test_incident_beam():
#    assert set(conversions.incident_beam().keys()) == set(['incident_beam'])
#
#
#def test_scattered_beam():
#    assert set(conversions.scattered_beam().keys()) == set(['scattered_beam'])
#
#
#def test_two_theta():
#    assert set(conversions.two_theta().keys()) == set(
#        ['scattered_beam', 'incident_beam', 'two_theta'])
#
#
#def test_L1():
#    assert set(conversions.L1().keys()) == set(['L1', 'incident_beam'])
#
#
#def test_L2():
#    assert set(conversions.L2().keys()) == set(['L2', 'scattered_beam'])
#
#
#def test_Ltotal():
#    assert set(conversions.Ltotal(scatter=False).keys()) == set(['Ltotal'])
#    assert set(conversions.Ltotal(scatter=True).keys()) == set(
#        ['scattered_beam', 'incident_beam', 'L1', 'L2', 'Ltotal'])
#
#
#def test_beamline():
#    assert set(conversions.beamline(scatter=False).keys()) == set(['Ltotal'])
#    assert set(conversions.beamline(scatter=True).keys()) == set(
#        ['scattered_beam', 'incident_beam', 'L1', 'L2', 'Ltotal', 'two_theta'])
#
#
#def test_kinematic():
#    assert set(conversions.kinematic('tof').keys()) == set(['energy', 'wavelength'])
#    # Other initial coords not supported for now
#    with pytest.raises(KeyError):
#        conversions.kinematic('energy')
#    with pytest.raises(KeyError):
#        conversions.kinematic('wavelength')
#
#
#def test_elastic():
#    assert set(conversions.elastic('energy').keys()) == set(['dspacing', 'wavelength'])
#    assert set(conversions.elastic('tof').keys()) == set(
#        ['dspacing', 'energy', 'Q', 'wavelength'])
#    assert set(conversions.elastic('Q').keys()) == set(['wavelength'])
#    assert set(conversions.elastic('wavelength').keys()) == set(
#        ['dspacing', 'energy', 'Q'])
#
#
#def test_elastic_dspacing():
#    assert set(conversions.elastic_dspacing('energy').keys()) == set(['dspacing'])
#    assert set(conversions.elastic_dspacing('tof').keys()) == set(['dspacing'])
#    assert set(conversions.elastic_dspacing('wavelength').keys()) == set(['dspacing'])
#
#
#def test_elastic_energy():
#    assert set(conversions.elastic_energy('tof').keys()) == set(['energy'])
#    assert set(conversions.elastic_energy('wavelength').keys()) == set(['energy'])
#
#
#def test_elastic_Q():
#    assert set(conversions.elastic_Q('tof').keys()) == set(['Q', 'wavelength'])
#    assert set(conversions.elastic_Q('wavelength').keys()) == set(['Q'])
#
#
#def test_elastic_wavelength():
#    assert set(conversions.elastic_wavelength('energy').keys()) == set(['wavelength'])
#    assert set(conversions.elastic_wavelength('tof').keys()) == set(['wavelength'])
#    assert set(conversions.elastic_wavelength('Q').keys()) == set(['wavelength'])
#
#
#def test_direct_inelastic():
#    assert set(conversions.direct_inelastic('tof').keys()) == set(['energy_transfer'])
#    # Other initial coords not supported for now
#    with pytest.raises(KeyError):
#        conversions.kinematic('wavelength')
#
#
#def test_indirect_inelastic():
#    assert set(conversions.indirect_inelastic('tof').keys()) == set(['energy_transfer'])
#    # Other initial coords not supported for now
#    with pytest.raises(KeyError):
#        conversions.kinematic('wavelength')
