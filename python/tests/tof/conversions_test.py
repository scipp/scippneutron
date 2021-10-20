# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
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
