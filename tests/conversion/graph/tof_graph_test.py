# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

import pytest

from scippneutron.conversion.graph import tof


def test_elastic():
    assert set(tof.elastic('energy').keys()) == {'dspacing', 'wavelength'}
    assert set(tof.elastic('tof').keys()) == {
        'dspacing',
        'energy',
        'Q',
        'wavelength',
        ('Qx', 'Qy', 'Qz'),
        'Q_vec',
        'hkl_vec',
        ('h', 'k', 'l'),
        'ub_matrix',
        'time_at_sample',
    }
    assert set(tof.elastic('Q').keys()) == {'wavelength'}
    assert set(tof.elastic('wavelength').keys()) == {
        'dspacing',
        'energy',
        'Q',
        ('Qx', 'Qy', 'Qz'),
        'Q_vec',
        'hkl_vec',
        ('h', 'k', 'l'),
        'ub_matrix',
    }


@pytest.mark.parametrize('start', ['dspacing'])
def test_elastic_unsupported_starts(start):
    with pytest.raises(KeyError):
        tof.elastic(start)


def test_kinematic():
    assert set(tof.kinematic('tof').keys()) == {'energy', 'wavelength'}
    assert set(tof.kinematic('wavelength').keys()) == {'energy'}
    assert set(tof.kinematic('energy').keys()) == {'wavelength'}


@pytest.mark.parametrize('start', ['dspacing', 'Q'])
def test_kinematic_unsupported_starts(start):
    with pytest.raises(KeyError):
        tof.kinematic(start)


def test_elastic_dspacing():
    assert set(tof.elastic_dspacing('energy').keys()) == {'dspacing'}
    assert set(tof.elastic_dspacing('tof').keys()) == {'dspacing'}
    assert set(tof.elastic_dspacing('wavelength').keys()) == {'dspacing'}


@pytest.mark.parametrize('start', ['Q'])
def test_elastic_dspacing_unsupported_starts(start):
    with pytest.raises(KeyError):
        tof.elastic_dspacing(start)


def test_elastic_energy():
    assert set(tof.elastic_energy('tof').keys()) == {'energy'}
    assert set(tof.elastic_energy('wavelength').keys()) == {'energy'}


@pytest.mark.parametrize('start', ['Q', 'dspacing'])
def test_elastic_energy_unsupported_starts(start):
    with pytest.raises(KeyError):
        tof.elastic_energy(start)


def test_elastic_Q():
    assert set(tof.elastic_Q('tof').keys()) == {'Q', 'wavelength'}
    assert set(tof.elastic_Q('wavelength').keys()) == {'Q'}


@pytest.mark.parametrize('start', ['energy', 'dspacing'])
def test_elastic_Q_unsupported_starts(start):
    with pytest.raises(KeyError):
        tof.elastic_Q(start)


def test_elastic_wavelength():
    assert set(tof.elastic_wavelength('energy').keys()) == {'wavelength'}
    assert set(tof.elastic_wavelength('tof').keys()) == {'wavelength'}
    assert set(tof.elastic_wavelength('Q').keys()) == {'wavelength'}


@pytest.mark.parametrize('start', ['dspacing'])
def test_elastic_wavelength_unsupported_starts(start):
    with pytest.raises(KeyError):
        tof.elastic_wavelength(start)


def test_direct_inelastic():
    assert set(tof.direct_inelastic('tof').keys()) == {'energy_transfer'}


@pytest.mark.parametrize('start', ['wavelength', 'Q', 'dspacing'])
def test_direct_inelastic_unsupported_starts(start):
    with pytest.raises(KeyError):
        tof.direct_inelastic(start)


def test_indirect_inelastic():
    assert set(tof.indirect_inelastic('tof').keys()) == {'energy_transfer'}


@pytest.mark.parametrize('start', ['wavelength', 'Q', 'dspacing'])
def test_indirect_inelastic_unsupported_starts(start):
    with pytest.raises(KeyError):
        tof.indirect_inelastic(start)


@pytest.mark.parametrize(
    'arg',
    [
        (tof.elastic, ('energy', 'tof', 'Q', 'wavelength')),
        (tof.kinematic, ('tof', 'wavelength', 'energy')),
        (tof.elastic_dspacing, ('tof', 'wavelength', 'energy')),
        (tof.elastic_energy, ('tof', 'wavelength')),
        (tof.elastic_Q, ('tof', 'wavelength')),
        (tof.elastic_wavelength, ('tof', 'energy', 'Q')),
        (tof.direct_inelastic, ('tof',)),
        (tof.indirect_inelastic, ('tof',)),
    ],
)
def test_returns_new_graph(arg):
    fn, starts = arg
    for start in starts:
        g = fn(start)
        g['a_new_node'] = lambda position: 2 * position
        assert 'a_new_node' not in fn(start)
