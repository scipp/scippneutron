# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

import numpy as np
import scipp as sc

from scippneutron.conversion import beamline


def test_straight_incident_beam():
    source_position = sc.vectors(dims=['siti'],
                                 values=[[40, 80, 20], [30, 10, 50]],
                                 unit='mm')
    sample_position = sc.vector([7, 5, 3], unit='mm')
    incident_beam = beamline.straight_incident_beam(source_position=source_position,
                                                    sample_position=sample_position)
    assert sc.allclose(
        incident_beam,
        sc.vectors(dims=['siti'], values=[[-33, -75, -17], [-23, -5, -47]], unit='mm'))


def test_straight_scattered_beam():
    position = sc.vectors(dims=['on'], values=[[6, 3, 9], [8, 3, 1]], unit='km')
    sample_position = sc.vector([0.4, 0.6, 0.2], unit='km')
    scattered_beam = beamline.straight_scattered_beam(position=position,
                                                      sample_position=sample_position)
    assert sc.allclose(
        scattered_beam,
        sc.vectors(dims=['on'], values=[[5.6, 2.4, 8.8], [7.6, 2.4, 0.8]], unit='km'))


def test_L1():
    incident_beam = sc.vectors(dims=['inc'],
                               values=[[0.5, 1.0, 1.5], [-0.3, 0.6, -0.9]],
                               unit='um')
    L1 = beamline.L1(incident_beam=incident_beam)
    assert sc.allclose(
        L1, sc.array(dims=['inc'], values=[1.870828693386, 1.122497216032], unit='um'))


def test_L2():
    scattered_beam = sc.vectors(dims=['scat'],
                                values=[[11, 22, 33], [95, 84, 73]],
                                unit='am')
    L2 = beamline.L2(scattered_beam=scattered_beam)
    assert sc.allclose(
        L2,
        sc.array(dims=['scat'], values=[41.158231254513, 146.321563687653], unit='am'))


def test_total_beam_length():
    L1 = sc.scalar(5.134, unit='cm')
    L2 = sc.array(dims=['secondary'], values=[3.14159, 42.0, 999.0], unit='cm')
    Ltotal = beamline.total_beam_length(L1=L1, L2=L2)
    assert sc.allclose(
        Ltotal,
        sc.array(dims=['secondary'], values=[8.27559, 47.134, 1004.134], unit='cm'))


def test_total_straight_beam_length_no_scatter():
    position = sc.vectors(dims=['po'], values=[[1, 2, 3], [4, 5, 6]], unit='m')
    source_position = sc.vector([0.1, 0.2, 0.3], unit='m')
    Ltotal = beamline.total_straight_beam_length_no_scatter(
        source_position=source_position, position=position)
    assert sc.allclose(
        Ltotal, sc.array(dims=['po'], values=[3.367491648096, 8.410707461325],
                         unit='m'))


def test_two_theta_arbitrary_values():
    incident_beam = sc.vector([0.564, 1.2, -10.4], unit='m')
    scattered_beam = sc.vectors(dims=['beam'],
                                values=[[13, 24, 35], [51, -42, 33]],
                                unit='m')
    two_theta = beamline.two_theta(incident_beam=incident_beam,
                                   scattered_beam=scattered_beam)
    assert sc.allclose(
        two_theta,
        sc.array(dims=['beam'], values=[2.352629742382, 2.061447408052], unit='rad'))


def test_two_theta_parallel_beams():
    incident_beam = sc.vector([1.0, 0.2, 0.0], unit='mm')
    scattered_beam = sc.array(dims=['beam'], values=[1.0, 2.0, 0.1]) * incident_beam
    two_theta = beamline.two_theta(incident_beam=incident_beam,
                                   scattered_beam=scattered_beam)
    assert sc.allclose(two_theta, sc.zeros(dims=['beam'], shape=[3], unit='rad'))


def test_two_theta_anti_parallel_beams():
    incident_beam = sc.vector([10.2, -0.8, 4.1], unit='mm')
    scattered_beam = sc.array(dims=['beam'], values=[-2.1, -31.0, -1.0]) * incident_beam
    two_theta = beamline.two_theta(incident_beam=incident_beam,
                                   scattered_beam=scattered_beam)
    assert sc.allclose(two_theta,
                       sc.full(value=np.pi, dims=['beam'], shape=[3], unit='rad'))


def test_two_theta_orthogonal_beams():
    incident_beam = sc.vector([0.1, 0.0, 10.0], unit='m')
    scattered_beam = sc.vectors(dims=['beam'],
                                values=[[0.0, -2.1, 0.0], [10.0, 123.0, -0.1]],
                                unit='m')
    two_theta = beamline.two_theta(incident_beam=incident_beam,
                                   scattered_beam=scattered_beam)
    assert sc.allclose(two_theta,
                       sc.full(value=np.pi / 2, dims=['beam'], shape=[2], unit='rad'))
