# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

import numpy as np
import pytest
import scipp as sc
import scipp.constants
import scipp.testing

from scippneutron.conversion import beamline


def test_straight_incident_beam():
    source_position = sc.vectors(
        dims=['siti'], values=[[40, 80, 20], [30, 10, 50]], unit='mm'
    )
    sample_position = sc.vector([7, 5, 3], unit='mm')
    incident_beam = beamline.straight_incident_beam(
        source_position=source_position, sample_position=sample_position
    )
    sc.testing.assert_allclose(
        incident_beam,
        sc.vectors(dims=['siti'], values=[[-33, -75, -17], [-23, -5, -47]], unit='mm'),
    )


def test_straight_scattered_beam():
    position = sc.vectors(dims=['on'], values=[[6, 3, 9], [8, 3, 1]], unit='km')
    sample_position = sc.vector([0.4, 0.6, 0.2], unit='km')
    scattered_beam = beamline.straight_scattered_beam(
        position=position, sample_position=sample_position
    )
    sc.testing.assert_allclose(
        scattered_beam,
        sc.vectors(dims=['on'], values=[[5.6, 2.4, 8.8], [7.6, 2.4, 0.8]], unit='km'),
    )


def test_L1():
    incident_beam = sc.vectors(
        dims=['inc'], values=[[0.5, 1.0, 1.5], [-0.3, 0.6, -0.9]], unit='um'
    )
    L1 = beamline.L1(incident_beam=incident_beam)
    sc.testing.assert_allclose(
        L1, sc.array(dims=['inc'], values=[1.870828693386, 1.122497216032], unit='um')
    )


def test_L2():
    scattered_beam = sc.vectors(
        dims=['scat'], values=[[11, 22, 33], [95, 84, 73]], unit='am'
    )
    L2 = beamline.L2(scattered_beam=scattered_beam)
    sc.testing.assert_allclose(
        L2,
        sc.array(dims=['scat'], values=[41.158231254513, 146.321563687653], unit='am'),
    )


def test_total_beam_length():
    L1 = sc.scalar(5.134, unit='cm')
    L2 = sc.array(dims=['secondary'], values=[3.14159, 42.0, 999.0], unit='cm')
    Ltotal = beamline.total_beam_length(L1=L1, L2=L2)
    sc.testing.assert_allclose(
        Ltotal,
        sc.array(dims=['secondary'], values=[8.27559, 47.134, 1004.134], unit='cm'),
    )


def test_total_straight_beam_length_no_scatter():
    position = sc.vectors(dims=['po'], values=[[1, 2, 3], [4, 5, 6]], unit='m')
    source_position = sc.vector([0.1, 0.2, 0.3], unit='m')
    Ltotal = beamline.total_straight_beam_length_no_scatter(
        source_position=source_position, position=position
    )
    sc.testing.assert_allclose(
        Ltotal, sc.array(dims=['po'], values=[3.367491648096, 8.410707461325], unit='m')
    )


def test_two_theta_arbitrary_values():
    incident_beam = sc.vector([0.564, 1.2, -10.4], unit='m')
    scattered_beam = sc.vectors(
        dims=['beam'], values=[[13, 24, 35], [51, -42, 33]], unit='m'
    )
    two_theta = beamline.two_theta(
        incident_beam=incident_beam, scattered_beam=scattered_beam
    )
    sc.testing.assert_allclose(
        two_theta,
        sc.array(dims=['beam'], values=[2.352629742382, 2.061447408052], unit='rad'),
    )


# TODO:
#   two_theta depends on the direction of incident_beam
#   scattering_angles_with_gravity does not


#   - check ranges of both [-pi/2, pi/2] or [0, pi] or something else?
#   - Which definition do we want? I think probably direction-dependent
#   - Ensure both are consistent!
def test_two_theta_depends_on_beam_direction():
    incident_beam = sc.vector([0.564, 1.2, -10.4], unit='m')
    scattered_beam = sc.vectors(
        dims=['beam'], values=[[13, 24, 35], [51, -42, 33]], unit='m'
    )
    a = beamline.two_theta(incident_beam=incident_beam, scattered_beam=scattered_beam)
    b = beamline.two_theta(incident_beam=-incident_beam, scattered_beam=scattered_beam)
    sc.testing.assert_allclose(
        a,
        sc.scalar(np.pi, unit='rad') - b,
    )


def test_two_theta_parallel_beams():
    incident_beam = sc.vector([1.0, 0.2, 0.0], unit='mm')
    scattered_beam = sc.array(dims=['beam'], values=[1.0, 2.0, 0.1]) * incident_beam
    two_theta = beamline.two_theta(
        incident_beam=incident_beam, scattered_beam=scattered_beam
    )
    sc.testing.assert_allclose(
        two_theta,
        sc.zeros(dims=['beam'], shape=[3], unit='rad'),
        atol=sc.scalar(1e-14, unit='rad'),
    )


def test_two_theta_anti_parallel_beams():
    incident_beam = sc.vector([10.2, -0.8, 4.1], unit='mm')
    scattered_beam = sc.array(dims=['beam'], values=[-2.1, -31.0, -1.0]) * incident_beam
    two_theta = beamline.two_theta(
        incident_beam=incident_beam, scattered_beam=scattered_beam
    )
    sc.testing.assert_allclose(
        two_theta, sc.full(value=np.pi, dims=['beam'], shape=[3], unit='rad')
    )


def test_two_theta_orthogonal_beams():
    incident_beam = sc.vector([0.1, 0.0, 10.0], unit='m')
    scattered_beam = sc.vectors(
        dims=['beam'], values=[[0.0, -2.1, 0.0], [10.0, 123.0, -0.1]], unit='m'
    )
    two_theta = beamline.two_theta(
        incident_beam=incident_beam, scattered_beam=scattered_beam
    )
    sc.testing.assert_allclose(
        two_theta, sc.full(value=np.pi / 2, dims=['beam'], shape=[2], unit='rad')
    )


def test_scattering_angles_requires_gravity_orthogonal_to_incident_beam():
    incident_beam = sc.vector([0.564, 1.2, -10.4], unit='m')
    scattered_beam = sc.vectors(
        dims=['beam'], values=[[13, 24, 35], [51, -42, 33]], unit='m'
    )
    wavelength = sc.array(
        dims=['wavelength'], values=[1.2, 1.6, 1.8], unit='Å', dtype='float32'
    )
    gravity = sc.vector([0, 0, sc.constants.g.value], unit=sc.constants.g.unit)

    with pytest.raises(
        ValueError, match='`gravity` and `incident_beam` must be orthogonal'
    ):
        beamline.scattering_angles_with_gravity(
            incident_beam=incident_beam,
            scattered_beam=scattered_beam,
            wavelength=wavelength,
            gravity=gravity,
        )


def test_scattering_angles_with_gravity_small_gravity():
    # This case is unphysical but tests the consistency of the implementation.
    incident_beam = sc.vector([0.564, 0.0, 10.4], unit='m')
    scattered_beam = sc.vectors(
        dims=['beam'], values=[[13, 24, 35], [51, -42, 33]], unit='m'
    )
    wavelength = sc.array(
        dims=['wavelength'], values=[1.2, 1.6, 1.8], unit='Å', dtype='float32'
    )
    gravity = sc.vector([0, -1e-11, 0], unit=sc.constants.g.unit)

    res = beamline.scattering_angles_with_gravity(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )
    expected = beamline.two_theta(
        incident_beam=incident_beam, scattered_beam=scattered_beam
    ).broadcast(dims=['beam', 'wavelength'], shape=[2, 3])
    sc.testing.assert_allclose(res['two_theta'], expected)
