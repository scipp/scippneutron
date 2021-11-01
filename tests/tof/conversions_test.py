# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import pytest
from scippneutron.tof import conversions


def test_incident_beam():
    assert set(conversions.incident_beam().keys()) == set(['incident_beam'])


def test_scattered_beam():
    assert set(conversions.scattered_beam().keys()) == set(['scattered_beam'])


def test_two_theta():
    assert set(conversions.two_theta().keys()) == set(
        ['scattered_beam', 'incident_beam', 'two_theta'])


def test_L1():
    assert set(conversions.L1().keys()) == set(['L1', 'incident_beam'])


def test_L2():
    assert set(conversions.L2().keys()) == set(['L2', 'scattered_beam'])


def test_Ltotal():
    assert set(conversions.Ltotal(scatter=False).keys()) == set(['Ltotal'])
    assert set(conversions.Ltotal(scatter=True).keys()) == set(
        ['scattered_beam', 'incident_beam', 'L1', 'L2', 'Ltotal'])


def test_beamline():
    assert set(conversions.beamline(scatter=False).keys()) == set(['Ltotal'])
    assert set(conversions.beamline(scatter=True).keys()) == set(
        ['scattered_beam', 'incident_beam', 'L1', 'L2', 'Ltotal', 'two_theta'])


def test_kinematic():
    assert set(conversions.kinematic('tof').keys()) == set(['energy', 'wavelength'])
    # Other initial coords not supported for now
    with pytest.raises(KeyError):
        conversions.kinematic('energy')
    with pytest.raises(KeyError):
        conversions.kinematic('wavelength')


def test_elastic():
    assert set(conversions.elastic('energy').keys()) == set(['dspacing', 'wavelength'])
    assert set(conversions.elastic('tof').keys()) == set(
        ['dspacing', 'energy', 'Q', 'wavelength'])
    assert set(conversions.elastic('Q').keys()) == set(['wavelength'])
    assert set(conversions.elastic('wavelength').keys()) == set(
        ['dspacing', 'energy', 'Q'])


def test_elastic_dspacing():
    assert set(conversions.elastic_dspacing('energy').keys()) == set(['dspacing'])
    assert set(conversions.elastic_dspacing('tof').keys()) == set(['dspacing'])
    assert set(conversions.elastic_dspacing('wavelength').keys()) == set(['dspacing'])


def test_elastic_energy():
    assert set(conversions.elastic_energy('tof').keys()) == set(['energy'])
    assert set(conversions.elastic_energy('wavelength').keys()) == set(['energy'])


def test_elastic_Q():
    assert set(conversions.elastic_Q('tof').keys()) == set(['Q', 'wavelength'])
    assert set(conversions.elastic_Q('wavelength').keys()) == set(['Q'])


def test_elastic_wavelength():
    assert set(conversions.elastic_wavelength('energy').keys()) == set(['wavelength'])
    assert set(conversions.elastic_wavelength('tof').keys()) == set(['wavelength'])
    assert set(conversions.elastic_wavelength('Q').keys()) == set(['wavelength'])


def test_direct_inelastic():
    assert set(conversions.direct_inelastic('tof').keys()) == set(['energy_transfer'])
    # Other initial coords not supported for now
    with pytest.raises(KeyError):
        conversions.kinematic('wavelength')


def test_indirect_inelastic():
    assert set(conversions.indirect_inelastic('tof').keys()) == set(['energy_transfer'])
    # Other initial coords not supported for now
    with pytest.raises(KeyError):
        conversions.kinematic('wavelength')
