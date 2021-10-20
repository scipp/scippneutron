# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import pytest
from scippneutron.tof import conversions


def test_incident_beam():
    graph = conversions.incident_beam()
    assert 'incident_beam' in graph
    assert len(graph) == 1


def test_scattered_beam():
    graph = conversions.scattered_beam()
    assert 'scattered_beam' in graph
    assert len(graph) == 1


def test_two_theta():
    graph = conversions.two_theta()
    assert 'two_theta' in graph
    assert len(graph) == 3


def test_L1():
    graph = conversions.L1()
    assert 'L1' in graph
    assert len(graph) == 2


def test_L2():
    graph = conversions.L2()
    assert 'L2' in graph
    assert len(graph) == 2


def test_Ltotal():
    graph = conversions.Ltotal(scatter=False)
    assert 'Ltotal' in graph
    assert len(graph) == 1
    graph = conversions.Ltotal(scatter=True)
    assert 'Ltotal' in graph
    assert len(graph) == 5


def test_beamline():
    assert len(conversions.beamline(scatter=False)) == 1
    assert len(conversions.beamline(scatter=True)) == 6


def test_kinematic():
    assert len(conversions.kinematic('tof')) == 2
    # Other initial coords not supported for now
    with pytest.raises(KeyError):
        conversions.kinematic('energy')
    with pytest.raises(KeyError):
        conversions.kinematic('wavelength')


def test_elastic():
    assert len(conversions.elastic('energy')) == 2
    assert len(conversions.elastic('tof')) == 4
    assert len(conversions.elastic('Q')) == 1
    assert len(conversions.elastic('wavelength')) == 3


def test_elastic_dspacing():
    graph = conversions.elastic_dspacing('energy')
    assert 'dspacing' in graph
    assert len(graph) == 1
    graph = conversions.elastic_dspacing('tof')
    assert 'dspacing' in graph
    assert len(graph) == 1
    graph = conversions.elastic_dspacing('wavelength')
    assert 'dspacing' in graph
    assert len(graph) == 1


def test_elastic_energy():
    graph = conversions.elastic_energy('tof')
    assert 'energy' in graph
    assert len(graph) == 1
    graph = conversions.elastic_energy('wavelength')
    assert 'energy' in graph
    assert len(graph) == 1


def test_elastic_Q():
    graph = conversions.elastic_Q('tof')
    assert 'Q' in graph
    assert len(graph) == 2
    graph = conversions.elastic_Q('wavelength')
    assert 'Q' in graph
    assert len(graph) == 1


def test_elastic_wavelength():
    graph = conversions.elastic_wavelength('energy')
    assert 'wavelength' in graph
    assert len(graph) == 1
    graph = conversions.elastic_wavelength('tof')
    assert 'wavelength' in graph
    assert len(graph) == 1
    graph = conversions.elastic_wavelength('Q')
    assert 'wavelength' in graph
    assert len(graph) == 1


def test_direct_inelastic():
    assert len(conversions.direct_inelastic('tof')) == 1
    # Other initial coords not supported for now
    with pytest.raises(KeyError):
        conversions.kinematic('wavelength')


def test_indirect_inelastic():
    assert len(conversions.indirect_inelastic('tof')) == 1
    # Other initial coords not supported for now
    with pytest.raises(KeyError):
        conversions.kinematic('wavelength')
