# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

import pytest

from scippneutron.conversion.graph import beamline


def test_incident_beam_has_correct_keys():
    assert set(beamline.incident_beam().keys()) == {'incident_beam'}


def test_scattered_beam_has_correct_keys():
    assert set(beamline.scattered_beam().keys()) == {'scattered_beam'}


def test_two_theta_has_correct_keys():
    assert set(beamline.two_theta().keys()) == {
        'scattered_beam',
        'incident_beam',
        'two_theta',
    }


def test_L1_has_correct_keys():
    assert set(beamline.L1().keys()) == {'L1', 'incident_beam'}


def test_L2_has_correct_keys():
    assert set(beamline.L2().keys()) == {'L2', 'scattered_beam'}


def test_Ltotal_has_correct_keys():
    assert set(beamline.Ltotal(scatter=False).keys()) == {'Ltotal'}
    assert set(beamline.Ltotal(scatter=True).keys()) == {
        'scattered_beam',
        'incident_beam',
        'L1',
        'L2',
        'Ltotal',
    }


def test_beamline_has_correct_keys():
    assert set(beamline.beamline(scatter=False).keys()) == {'Ltotal'}
    assert set(beamline.beamline(scatter=True).keys()) == {
        'scattered_beam',
        'incident_beam',
        'L1',
        'L2',
        'Ltotal',
        'two_theta',
    }


@pytest.mark.parametrize(
    'fn',
    [
        beamline.incident_beam,
        beamline.scattered_beam,
        beamline.L1,
        beamline.L2,
        beamline.two_theta,
    ],
)
def test_beamline_returns_new_graph_without_scatter_arg(fn):
    g = fn()
    g['a_new_node'] = lambda position: 2 * position
    assert 'a_new_node' not in fn()


@pytest.mark.parametrize('fn', [beamline.beamline, beamline.Ltotal])
@pytest.mark.parametrize('scatter', [True, False])
def test_beamline_returns_new_graph_with_scatter_arg(fn, scatter):
    g = fn(scatter=scatter)
    g['a_new_node'] = lambda position: 2 * position
    assert 'a_new_node' not in fn(scatter=scatter)
