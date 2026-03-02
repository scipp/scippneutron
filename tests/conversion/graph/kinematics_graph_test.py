# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

import pytest

from scippneutron.conversion.graph import kinematics


def test_elastic():
    assert set(kinematics.elastic('energy').keys()) == {'dspacing', 'wavelength'}
    assert set(kinematics.elastic('wavelength').keys()) == {
        'dspacing',
        'energy',
        'Q',
        'tof',
        ('Qx', 'Qy', 'Qz'),
        'Q_vec',
        'hkl_vec',
        ('h', 'k', 'l'),
        'ub_matrix',
        'time_at_sample',
    }
    assert set(kinematics.elastic('Q').keys()) == {'wavelength'}


@pytest.mark.parametrize('start', ['dspacing'])
def test_elastic_unsupported_starts(start):
    with pytest.raises(KeyError):
        kinematics.elastic(start)


def test_kinematic():
    assert set(kinematics.kinematic('wavelength').keys()) == {'energy'}
    assert set(kinematics.kinematic('energy').keys()) == {'wavelength'}


@pytest.mark.parametrize('start', ['dspacing', 'Q'])
def test_kinematic_unsupported_starts(start):
    with pytest.raises(KeyError):
        kinematics.kinematic(start)


def test_elastic_dspacing():
    assert set(kinematics.elastic_dspacing('energy').keys()) == {'dspacing'}
    assert set(kinematics.elastic_dspacing('wavelength').keys()) == {'dspacing'}


@pytest.mark.parametrize('start', ['Q'])
def test_elastic_dspacing_unsupported_starts(start):
    with pytest.raises(KeyError):
        kinematics.elastic_dspacing(start)


def test_elastic_energy():
    assert set(kinematics.elastic_energy('wavelength').keys()) == {'energy'}


@pytest.mark.parametrize('start', ['Q', 'dspacing'])
def test_elastic_energy_unsupported_starts(start):
    with pytest.raises(KeyError):
        kinematics.elastic_energy(start)


def test_elastic_Q():
    assert set(kinematics.elastic_Q('wavelength').keys()) == {'Q'}


@pytest.mark.parametrize('start', ['energy', 'dspacing'])
def test_elastic_Q_unsupported_starts(start):
    with pytest.raises(KeyError):
        kinematics.elastic_Q(start)


def test_elastic_tof():
    assert set(kinematics.elastic_tof('wavelength').keys()) == {'tof'}


@pytest.mark.parametrize('start', ['dspacing'])
def test_elastic_tof_unsupported_starts(start):
    with pytest.raises(KeyError):
        kinematics.elastic_tof(start)


def test_direct_inelastic():
    assert set(kinematics.direct_inelastic('wavelength').keys()) == {'energy_transfer'}


@pytest.mark.parametrize('start', ['tof', 'Q', 'dspacing'])
def test_direct_inelastic_unsupported_starts(start):
    with pytest.raises(KeyError):
        kinematics.direct_inelastic(start)


def test_indirect_inelastic():
    assert set(kinematics.indirect_inelastic('wavelength').keys()) == {
        'energy_transfer'
    }


@pytest.mark.parametrize('start', ['tof', 'Q', 'dspacing'])
def test_indirect_inelastic_unsupported_starts(start):
    with pytest.raises(KeyError):
        kinematics.indirect_inelastic(start)


@pytest.mark.parametrize(
    'arg',
    [
        (kinematics.elastic, ('energy', 'Q', 'wavelength')),
        (kinematics.kinematic, ('wavelength', 'energy')),
        (kinematics.elastic_dspacing, ('wavelength', 'energy')),
        (kinematics.elastic_energy, ('wavelength', 'energy')),
        (kinematics.elastic_Q, ('wavelength', 'energy')),
        (kinematics.elastic_tof, ('wavelength', 'energy', 'Q')),
        (kinematics.direct_inelastic, ('wavelength',)),
        (kinematics.indirect_inelastic, ('wavelength',)),
    ],
)
def test_returns_new_graph(arg):
    fn, starts = arg
    for start in starts:
        g = fn(start)
        g['a_new_node'] = lambda position: 2 * position
        assert 'a_new_node' not in fn(start)
