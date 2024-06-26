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


def test_two_theta_invariant_under_reflection_about_incident_beam():
    incident_beam = sc.vector([0.564, 1.2, -10.4], unit='cm')
    rotation = sc.spatial.rotations_from_rotvecs(
        sc.vectors(
            dims=['scattered'], values=[[0.7, 0.0, 0.0], [-0.7, 0.0, 0.0]], unit='rad'
        )
    )
    scattered_beam = rotation * incident_beam
    two_theta = beamline.two_theta(
        incident_beam=incident_beam, scattered_beam=scattered_beam
    )
    sc.testing.assert_allclose(two_theta[0], two_theta[1])


def test_scattering_angles_requires_gravity_orthogonal_to_incident_beam():
    incident_beam = sc.vector([0.564, 1.2, -10.4], unit='m')
    scattered_beam = sc.vectors(
        dims=['beam'], values=[[13, 24, 35], [51, -42, 33]], unit='m'
    )
    wavelength = sc.array(dims=['wavelength'], values=[1.2, 1.6, 1.8], unit='Å')
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
    # This case is unphysical but tests the consistency with `two_theta`.
    incident_beam = sc.vector([0.564, 0.0, 10.4], unit='m')
    scattered_beam = sc.vectors(
        dims=['beam'],
        values=[[13, 24, 35], [51, -42, 33], [4, 23, -17], [-19, -31, 5]],
        unit='m',
    )
    wavelength = sc.array(dims=['wavelength'], values=[1.2, 1.6, 1.8], unit='Å')
    gravity = sc.vector([0, -1e-11, 0], unit=sc.constants.g.unit)

    res = beamline.scattering_angles_with_gravity(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )
    expected = beamline.two_theta(
        incident_beam=incident_beam, scattered_beam=scattered_beam
    ).broadcast(dims=['wavelength', 'beam'], shape=[3, 4])
    sc.testing.assert_allclose(res['two_theta'], expected)


@pytest.mark.parametrize('polar', [np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi])
@pytest.mark.parametrize('azimuthal', [0.0, np.pi / 2, np.pi])
def test_scattering_angles_with_gravity_reproduces_angles(
    polar: float, azimuthal: float
):
    # This case is unphysical but tests that the function reproduces
    # the expected angles using a rotated vector.

    gravity = sc.vector([0.0, -1e-11, 0.0], unit='cm/s^2')
    incident_beam = sc.vector([0.0, 0.0, 968.0], unit='cm')

    # With this definition, the x-axis has azimuthal=0.
    rot1 = sc.spatial.rotations_from_rotvecs(sc.vector([-polar, 0, 0], unit='rad'))
    rot2 = sc.spatial.rotations_from_rotvecs(
        sc.vector([0, 0, azimuthal - np.pi / 2], unit='rad')
    )
    scattered_beam = rot2 * (rot1 * incident_beam)

    wavelength = sc.scalar(1e-6, unit='Å')

    res = beamline.scattering_angles_with_gravity(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )

    sc.testing.assert_allclose(
        res['two_theta'],
        sc.scalar(polar, unit='rad'),
        atol=sc.scalar(1e-15, unit='rad'),
    )
    sc.testing.assert_allclose(
        res['phi'], sc.scalar(azimuthal, unit='rad'), atol=sc.scalar(1e-15, unit='rad')
    )


@pytest.mark.parametrize('polar', [np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi])
@pytest.mark.parametrize('azimuthal', [3 * np.pi / 2, 8 * np.pi / 5, 2 * np.pi])
def test_scattering_angles_with_gravity_reproduces_angles_azimuth_greater_pi(
    polar: float, azimuthal: float
):
    # This case is unphysical but tests that the function reproduces
    # the expected angles using a rotated vector.

    gravity = sc.vector([0.0, -1e-11, 0.0], unit='cm/s^2')
    incident_beam = sc.vector([0.0, 0.0, 968.0], unit='cm')

    # With this definition, the x-axis has azimuthal=0.
    rot1 = sc.spatial.rotations_from_rotvecs(sc.vector([-polar, 0, 0], unit='rad'))
    rot2 = sc.spatial.rotations_from_rotvecs(
        sc.vector([0, 0, azimuthal - np.pi / 2], unit='rad')
    )
    scattered_beam = rot2 * (rot1 * incident_beam)

    wavelength = sc.scalar(1e-6, unit='Å')

    res = beamline.scattering_angles_with_gravity(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )

    sc.testing.assert_allclose(
        res['two_theta'],
        sc.scalar(polar, unit='rad'),
        atol=sc.scalar(1e-15, unit='rad'),
    )
    # phi has range (-pi, pi], so for azimuthal > pi, we get a negative result
    sc.testing.assert_allclose(
        res['phi'],
        sc.scalar(azimuthal - 2 * np.pi, unit='rad'),
        atol=sc.scalar(1e-15, unit='rad'),
    )


@pytest.mark.parametrize('azimuthal', [0.0, np.pi / 2, np.pi, 7 * np.pi / 5, 2 * np.pi])
def test_scattering_angles_with_gravity_reproduces_angles_polar_zero(azimuthal: float):
    # This case is unphysical but tests that the function reproduces
    # the expected angles using a rotated vector.

    gravity = sc.vector([0.0, -1e-11, 0.0], unit='cm/s^2')
    incident_beam = sc.vector([0.0, 0.0, 968.0], unit='cm')

    # With this definition, the x-axis has azimuthal=0.
    rot = sc.spatial.rotations_from_rotvecs(
        sc.vector([0, 0, azimuthal - np.pi / 2], unit='rad')
    )
    scattered_beam = rot * incident_beam

    wavelength = sc.scalar(1e-6, unit='Å')

    res = beamline.scattering_angles_with_gravity(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )

    sc.testing.assert_allclose(
        res['two_theta'], sc.scalar(0.0, unit='rad'), atol=sc.scalar(1e-15, unit='rad')
    )
    # When polar = 0, the azimuthal angle is ill-defined, so there is no test.


def test_scattering_angles_with_gravity_drops_in_expected_direction():
    wavelength = sc.scalar(1.6, unit='Å')
    gravity = sc.vector([0.0, -sc.constants.g.value, 0.0], unit=sc.constants.g.unit)
    incident_beam = sc.vector([0.0, 0.0, 41.1], unit='m')
    scattered_beam = sc.vectors(
        dims=['det'], values=[[0.0, 2.5, 8.6], [0.0, -1.7, 6.9]], unit='m'
    )

    with_gravity = beamline.scattering_angles_with_gravity(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )
    without_gravity = beamline.two_theta(
        incident_beam=incident_beam, scattered_beam=scattered_beam
    )

    # The neutron was detected above the incident beam.
    # So using straight paths, it looks like it scattered at a
    # smaller angle (detected at a lower y) than in the real case with gravity.
    assert sc.all(with_gravity['two_theta'][0] > without_gravity[0]).value
    # The neutron was detected below the incident beam.
    # So the opposite of the above comment applies.
    assert sc.all(with_gravity['two_theta'][1] < without_gravity[1]).value
    sc.testing.assert_allclose(
        with_gravity['phi'],
        sc.array(dims=['det'], values=[np.pi / 2, -np.pi / 2], unit='rad'),
    )


def test_scattering_angles_with_gravity_beams_aligned_with_lab_coords():
    wavelength = sc.array(dims=['wavelength'], values=[1.6, 0.9, 0.7], unit='Å')
    gravity = sc.vector([0.0, -sc.constants.g.value, 0.0], unit=sc.constants.g.unit)
    incident_beam = sc.vector([0.0, 0.0, 41.1], unit='m')
    scattered_beam = sc.vectors(
        dims=['det'], values=[[0.0, 2.5, 3.6], [0.0, -1.7, 2.9]], unit='m'
    )

    original_wavelength = wavelength.copy()
    original_gravity = gravity.copy()
    original_incident_beam = incident_beam.copy()
    original_scattered_beam = scattered_beam.copy()

    res = beamline.scattering_angles_with_gravity(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )

    L2 = sc.norm(scattered_beam)
    x = scattered_beam.fields.x
    y = scattered_beam.fields.y
    z = scattered_beam.fields.z
    drop = (
        L2**2
        * sc.norm(gravity)
        * wavelength**2
        * sc.constants.m_n**2
        / (2 * sc.constants.h**2)
    )
    drop = drop.to(unit=y.unit)
    true_y = y + drop
    true_scattered_beam = sc.spatial.as_vectors(x, true_y, z)
    expected_two_theta = sc.asin(
        sc.sqrt(x**2 + true_y**2) / sc.norm(true_scattered_beam)
    )
    expected_two_theta = expected_two_theta.transpose(res['two_theta'].dims)

    expected_phi = sc.atan2(y=true_y, x=x).transpose(res['phi'].dims)

    sc.testing.assert_allclose(res['two_theta'], expected_two_theta)
    sc.testing.assert_allclose(res['phi'], expected_phi)

    sc.testing.assert_identical(wavelength, original_wavelength)
    sc.testing.assert_identical(gravity, original_gravity)
    sc.testing.assert_identical(incident_beam, original_incident_beam)
    sc.testing.assert_identical(scattered_beam, original_scattered_beam)


def _reference_scattering_angles_with_gravity(
    incident_beam: sc.Variable,
    scattered_beam: sc.Variable,
    gravity: sc.Variable,
    wavelength: sc.Variable,
) -> dict[str, sc.Variable]:
    # This is a simplified, independently checked implementation.
    e_z = incident_beam / sc.norm(incident_beam)
    e_y = -gravity / sc.norm(gravity)
    e_x = sc.cross(e_y, e_z)

    x = sc.dot(scattered_beam, e_x)
    y = sc.dot(scattered_beam, e_y)
    z = sc.dot(scattered_beam, e_z)

    L2 = sc.norm(scattered_beam)
    drop = (
        L2**2
        * wavelength**2
        * sc.norm(gravity)
        * (sc.constants.m_n**2 / (2 * sc.constants.h**2))
    )
    dropped_y = y + drop.to(unit=y.unit)

    two_theta = sc.atan2(y=sc.sqrt(x**2 + dropped_y**2), x=z)
    phi = sc.atan2(y=dropped_y, x=x)
    return {'two_theta': two_theta, 'phi': phi}


def test_scattering_angles_with_gravity_beams_unaligned_with_lab_coords():
    wavelength = sc.array(dims=['wavelength'], values=[1.6, 0.9, 0.7], unit='Å')
    # Gravity and incident_beam are not aligned with the coordinate system
    # but orthogonal to each other.
    gravity = sc.vector([-0.3, -9.81, 0.01167883211678832], unit='m/s^2')
    incident_beam = sc.vector([1.6, 0.0, 41.1], unit='m')
    scattered_beam = sc.vectors(
        dims=['det'], values=[[1.8, 2.5, 3.6], [-0.4, -1.7, 2.9]], unit='m'
    )

    original_wavelength = wavelength.copy()
    original_gravity = gravity.copy()
    original_incident_beam = incident_beam.copy()
    original_scattered_beam = scattered_beam.copy()

    res = beamline.scattering_angles_with_gravity(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )
    expected = _reference_scattering_angles_with_gravity(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )

    sc.testing.assert_allclose(
        res['two_theta'],
        expected['two_theta'].transpose(res['two_theta'].dims),
    )
    sc.testing.assert_allclose(
        res['phi'],
        expected['phi'].transpose(res['phi'].dims),
    )

    sc.testing.assert_identical(wavelength, original_wavelength)
    sc.testing.assert_identical(gravity, original_gravity)
    sc.testing.assert_identical(incident_beam, original_incident_beam)
    sc.testing.assert_identical(scattered_beam, original_scattered_beam)


def test_scattering_angles_with_gravity_binned_data():
    wavelength = sc.array(dims=['wavelength'], values=[1.6, 0.9, 0.7], unit='Å')
    wavelength = sc.bins(
        dim='wavelength',
        data=wavelength,
        begin=sc.array(dims=['det'], values=[0, 2], unit=None),
        end=sc.array(dims=['det'], values=[2, 3], unit=None),
    )
    gravity = sc.vector([-0.3, -9.81, 0.01167883211678832], unit='m/s^2')
    incident_beam = sc.vector([1.6, 0.0, 41.1], unit='m')
    scattered_beam = sc.vectors(
        dims=['det'], values=[[1.8, 2.5, 3.6], [-0.4, -1.7, 2.9]], unit='m'
    )

    original_wavelength = wavelength.copy()
    original_gravity = gravity.copy()
    original_incident_beam = incident_beam.copy()
    original_scattered_beam = scattered_beam.copy()

    res = beamline.scattering_angles_with_gravity(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )
    expected = _reference_scattering_angles_with_gravity(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )

    sc.testing.assert_allclose(
        res['two_theta'],
        expected['two_theta'].transpose(res['two_theta'].dims),
    )
    sc.testing.assert_allclose(res['phi'], expected['phi'].transpose(res['phi'].dims))

    sc.testing.assert_identical(wavelength, original_wavelength)
    sc.testing.assert_identical(gravity, original_gravity)
    sc.testing.assert_identical(incident_beam, original_incident_beam)
    sc.testing.assert_identical(scattered_beam, original_scattered_beam)


def test_scattering_angles_with_gravity_uses_wavelength_dtype():
    wavelength = sc.array(
        dims=['wavelength'], values=[1.6, 0.7], unit='Å', dtype='float32'
    )
    gravity = sc.vector([0.0, -9.81, 0.0], unit='m/s^2')
    incident_beam = sc.vector([0.0, 0.0, 41.1], unit='m')
    scattered_beam = sc.vector([1.8, 2.5, 3.6], unit='m')

    res = beamline.scattering_angles_with_gravity(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )
    assert res['two_theta'].dtype == 'float32'
    assert res['phi'].dtype == 'float32'


def test_scattering_angles_with_gravity_supports_mismatching_units():
    wavelength = sc.array(dims=['wavelength'], values=[1.6, 0.7], unit='Å')
    gravity = sc.vector([0.0, -9810, 0.0], unit='m/ms^2')
    incident_beam = sc.vector([0.0, 0.0, 410], unit='cm')
    scattered_beam = sc.vector([180, 1800, 2400], unit='mm')

    res = beamline.scattering_angles_with_gravity(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )

    expected = _reference_scattering_angles_with_gravity(
        incident_beam=incident_beam.to(unit='m'),
        scattered_beam=scattered_beam.to(unit='m'),
        wavelength=wavelength,
        gravity=gravity.to(unit='m/s^2'),
    )

    sc.testing.assert_allclose(res['two_theta'], expected['two_theta'])
    sc.testing.assert_allclose(res['phi'], expected['phi'])


def test_scattering_angle_in_yz_plane_requires_gravity_orthogonal_to_incident_beam():
    incident_beam = sc.vector([0.564, 1.2, -10.4], unit='m')
    scattered_beam = sc.vectors(
        dims=['beam'], values=[[13, 24, 35], [51, -42, 33]], unit='m'
    )
    wavelength = sc.array(dims=['wavelength'], values=[1.2, 1.6, 1.8], unit='Å')
    gravity = sc.vector([0, 0, sc.constants.g.value], unit=sc.constants.g.unit)

    with pytest.raises(
        ValueError, match='`gravity` and `incident_beam` must be orthogonal'
    ):
        beamline.scattering_angle_in_yz_plane(
            incident_beam=incident_beam,
            scattered_beam=scattered_beam,
            wavelength=wavelength,
            gravity=gravity,
        )


def test_scattering_angle_in_yz_plane_small_gravity():
    # This case is unphysical but tests the consistency with `two_theta`.
    # Note that the scattered beam must be in the x-z plane for `two_theta`
    # and `scattering_angle_from_sample` to compute the same angle.
    incident_beam = sc.vector([0.0, 0.0, 10.4], unit='m')
    scattered_beam = sc.vectors(
        dims=['beam'],
        values=[[0, 24, 0], [0, -42, 0]],
        unit='m',
    )
    wavelength = sc.array(dims=['wavelength'], values=[1.2, 1.6, 1.8], unit='Å')
    gravity = sc.vector([0, -1e-11, 0], unit=sc.constants.g.unit)

    res = beamline.scattering_angle_in_yz_plane(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )
    expected = beamline.two_theta(
        incident_beam=incident_beam, scattered_beam=scattered_beam
    ).broadcast(dims=['wavelength', 'beam'], shape=[3, 2])
    sc.testing.assert_allclose(res, expected)


@pytest.mark.parametrize('polar', [np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi])
def test_scattering_angle_in_yz_plane_reproduces_polar_angle(
    polar: float,
):
    # This case is unphysical but tests that the function reproduces
    # the expected angles using a rotated vector.

    gravity = sc.vector([0.0, -1e-11, 0.0], unit='cm/s^2')
    incident_beam = sc.vector([0.0, 0.0, 968.0], unit='cm')

    # With this definition, the x-axis has azimuthal=0.
    rot1 = sc.spatial.rotations_from_rotvecs(sc.vector([-polar, 0, 0], unit='rad'))
    scattered_beam = rot1 * incident_beam

    wavelength = sc.scalar(1e-6, unit='Å')

    res = beamline.scattering_angle_in_yz_plane(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )

    sc.testing.assert_allclose(res, sc.scalar(polar, unit='rad'))


@pytest.mark.parametrize('polar', [np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi])
@pytest.mark.parametrize('x', [0.5, 11.4, -9.7])
def test_scattering_angle_in_yz_plane_does_not_depend_on_x(polar: float, x: float):
    # This case is unphysical but tests that the function reproduces
    # the expected angles using a rotated vector.

    gravity = sc.vector([0.0, -1e-11, 0.0], unit='cm/s^2')
    incident_beam = sc.vector([0.0, 0.0, 968.0], unit='cm')

    # With this definition, the x-axis has azimuthal=0.
    rot1 = sc.spatial.rotations_from_rotvecs(sc.vector([-polar, 0, 0], unit='rad'))
    scattered_beam_ref = rot1 * incident_beam
    scattered_beam_shift = scattered_beam_ref + sc.vector([x, 0.0, 0.0], unit='cm')

    wavelength = sc.scalar(1e-6, unit='Å')

    res_shift = beamline.scattering_angle_in_yz_plane(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam_shift,
        wavelength=wavelength,
        gravity=gravity,
    )
    res_ref = beamline.scattering_angle_in_yz_plane(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam_ref,
        wavelength=wavelength,
        gravity=gravity,
    )

    sc.testing.assert_allclose(res_shift, res_ref)


def test_scattering_angle_in_yz_plane_drops_in_expected_direction():
    wavelength = sc.scalar(1.6, unit='Å')
    gravity = sc.vector([0.0, -sc.constants.g.value, 0.0], unit=sc.constants.g.unit)
    incident_beam = sc.vector([0.0, 0.0, 41.1], unit='m')
    scattered_beam = sc.vectors(
        dims=['det'], values=[[0.0, 2.5, 8.6], [0.0, -1.7, 6.9]], unit='m'
    )

    with_gravity = beamline.scattering_angle_in_yz_plane(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )
    without_gravity = beamline.two_theta(
        incident_beam=incident_beam, scattered_beam=scattered_beam
    )

    # The neutron was detected above the incident beam.
    # So using straight paths, it looks like it scattered at a
    # smaller angle (detected at a lower y) than in the real case with gravity.
    assert sc.all(with_gravity[0] > without_gravity[0]).value
    # The neutron was detected below the incident beam.
    # So the opposite of the above comment applies.
    assert sc.all(with_gravity[1] < without_gravity[1]).value


def test_scattering_angle_in_yz_plane_beams_aligned_with_lab_coords():
    wavelength = sc.array(dims=['wavelength'], values=[1.6, 0.9, 0.7], unit='Å')
    gravity = sc.vector([0.0, -sc.constants.g.value, 0.0], unit=sc.constants.g.unit)
    incident_beam = sc.vector([0.0, 0.0, 41.1], unit='m')
    scattered_beam = sc.vectors(
        dims=['det'], values=[[0.0, 2.5, 3.6], [0.0, -1.7, 2.9]], unit='m'
    )

    original_wavelength = wavelength.copy()
    original_gravity = gravity.copy()
    original_incident_beam = incident_beam.copy()
    original_scattered_beam = scattered_beam.copy()

    res = beamline.scattering_angle_in_yz_plane(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )

    L2 = sc.norm(scattered_beam)
    x = scattered_beam.fields.x
    y = scattered_beam.fields.y
    z = scattered_beam.fields.z
    drop = (
        L2**2
        * sc.norm(gravity)
        * wavelength**2
        * sc.constants.m_n**2
        / (2 * sc.constants.h**2)
    )
    drop = drop.to(unit=y.unit)
    true_y = y + drop
    true_scattered_beam = sc.spatial.as_vectors(x, true_y, z)
    expected_theta = sc.asin(abs(true_y) / sc.norm(true_scattered_beam))
    expected_theta = expected_theta.transpose(res.dims)

    sc.testing.assert_allclose(res, expected_theta)

    sc.testing.assert_identical(wavelength, original_wavelength)
    sc.testing.assert_identical(gravity, original_gravity)
    sc.testing.assert_identical(incident_beam, original_incident_beam)
    sc.testing.assert_identical(scattered_beam, original_scattered_beam)


def _reference_scattering_angle_in_yz_plane(
    incident_beam: sc.Variable,
    scattered_beam: sc.Variable,
    gravity: sc.Variable,
    wavelength: sc.Variable,
) -> sc.Variable:
    # This is a simplified, independently checked implementation.
    e_z = incident_beam / sc.norm(incident_beam)
    e_y = -gravity / sc.norm(gravity)

    y = sc.dot(scattered_beam, e_y)
    z = sc.dot(scattered_beam, e_z)

    L2 = sc.norm(scattered_beam)
    drop = (
        L2**2
        * wavelength**2
        * sc.norm(gravity)
        * (sc.constants.m_n**2 / (2 * sc.constants.h**2))
    )
    dropped_y = y + drop.to(unit=y.unit)

    return sc.atan2(y=abs(dropped_y), x=z)


def test_scattering_angle_in_yz_plane_beams_unaligned_with_lab_coords():
    wavelength = sc.array(dims=['wavelength'], values=[1.6, 0.9, 0.7], unit='Å')
    # Gravity and incident_beam are not aligned with the coordinate system
    # but orthogonal to each other.
    gravity = sc.vector([-0.3, -9.81, 0.01167883211678832], unit='m/s^2')
    incident_beam = sc.vector([1.6, 0.0, 41.1], unit='m')
    scattered_beam = sc.vectors(
        dims=['det'], values=[[1.8, 2.5, 3.6], [-0.4, -1.7, 2.9]], unit='m'
    )

    original_wavelength = wavelength.copy()
    original_gravity = gravity.copy()
    original_incident_beam = incident_beam.copy()
    original_scattered_beam = scattered_beam.copy()

    res = beamline.scattering_angle_in_yz_plane(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )
    expected = _reference_scattering_angle_in_yz_plane(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )

    sc.testing.assert_allclose(res, expected.transpose(res.dims))

    sc.testing.assert_identical(wavelength, original_wavelength)
    sc.testing.assert_identical(gravity, original_gravity)
    sc.testing.assert_identical(incident_beam, original_incident_beam)
    sc.testing.assert_identical(scattered_beam, original_scattered_beam)


def test_scattering_angle_in_yz_plane_binned_data():
    wavelength = sc.array(dims=['wavelength'], values=[1.6, 0.9, 0.7], unit='Å')
    wavelength = sc.bins(
        dim='wavelength',
        data=wavelength,
        begin=sc.array(dims=['det'], values=[0, 2], unit=None),
        end=sc.array(dims=['det'], values=[2, 3], unit=None),
    )
    gravity = sc.vector([-0.3, -9.81, 0.01167883211678832], unit='m/s^2')
    incident_beam = sc.vector([1.6, 0.0, 41.1], unit='m')
    scattered_beam = sc.vectors(
        dims=['det'], values=[[1.8, 2.5, 3.6], [-0.4, -1.7, 2.9]], unit='m'
    )

    original_wavelength = wavelength.copy()
    original_gravity = gravity.copy()
    original_incident_beam = incident_beam.copy()
    original_scattered_beam = scattered_beam.copy()

    res = beamline.scattering_angle_in_yz_plane(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )
    expected = _reference_scattering_angle_in_yz_plane(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )

    sc.testing.assert_allclose(res, expected.transpose(res.dims))
    sc.testing.assert_identical(wavelength, original_wavelength)
    sc.testing.assert_identical(gravity, original_gravity)
    sc.testing.assert_identical(incident_beam, original_incident_beam)
    sc.testing.assert_identical(scattered_beam, original_scattered_beam)


def test_scattering_angle_in_yz_plane_uses_wavelength_dtype():
    wavelength = sc.array(
        dims=['wavelength'], values=[1.6, 0.7], unit='Å', dtype='float32'
    )
    gravity = sc.vector([0.0, -9.81, 0.0], unit='m/s^2')
    incident_beam = sc.vector([0.0, 0.0, 41.1], unit='m')
    scattered_beam = sc.vector([1.8, 2.5, 3.6], unit='m')

    res = beamline.scattering_angle_in_yz_plane(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )
    assert res.dtype == 'float32'


def test_scattering_angle_in_yz_plane_supports_mismatching_units():
    wavelength = sc.array(dims=['wavelength'], values=[1.6, 0.7], unit='Å')
    gravity = sc.vector([0.0, -9810, 0.0], unit='m/ms^2')
    incident_beam = sc.vector([0.0, 0.0, 410], unit='cm')
    scattered_beam = sc.vector([180, 1800, 2400], unit='mm')

    res = beamline.scattering_angle_in_yz_plane(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )

    expected = _reference_scattering_angle_in_yz_plane(
        incident_beam=incident_beam.to(unit='m'),
        scattered_beam=scattered_beam.to(unit='m'),
        wavelength=wavelength,
        gravity=gravity.to(unit='m/s^2'),
    )

    sc.testing.assert_allclose(res, expected)


def test_beam_aligned_unit_vectors_axis_aligned_inputs():
    res = beamline.beam_aligned_unit_vectors(
        incident_beam=sc.vector([0.0, 0.0, 2.1], unit='mm'),
        gravity=sc.vector([0.0, -4.6, 0.0], unit='m/s/s'),
    )
    assert len(res) == 3
    sc.testing.assert_identical(res['beam_aligned_unit_x'], sc.vector([1.0, 0.0, 0.0]))
    sc.testing.assert_identical(res['beam_aligned_unit_y'], sc.vector([0.0, 1.0, 0.0]))
    sc.testing.assert_identical(res['beam_aligned_unit_z'], sc.vector([0.0, 0.0, 1.0]))


def test_beam_aligned_unit_vectors_complicated_inputs():
    incident_beam = sc.vector([3.1, -0.2, 23.6], unit='m')
    gravity = sc.vector([-0.01, -9.5, 3.2], unit='m/s/s')
    # Subtract projection of gravity onto incident_beam to make the vectors orthogonal.
    gravity -= sc.to_unit(
        sc.dot(incident_beam, gravity) * incident_beam / sc.norm(incident_beam) ** 2,
        'm/s/s',
    )
    res = beamline.beam_aligned_unit_vectors(
        incident_beam=incident_beam, gravity=gravity
    )
    assert len(res) == 3
    ex = res['beam_aligned_unit_x']
    ey = res['beam_aligned_unit_y']
    ez = res['beam_aligned_unit_z']
    sc.testing.assert_allclose(sc.norm(ex), sc.scalar(1.0))
    sc.testing.assert_allclose(sc.norm(ey), sc.scalar(1.0))
    sc.testing.assert_allclose(sc.norm(ez), sc.scalar(1.0))
    sc.testing.assert_allclose(sc.dot(ex, ey), sc.scalar(0.0), atol=sc.scalar(1e-16))
    sc.testing.assert_allclose(sc.dot(ey, ez), sc.scalar(0.0), atol=sc.scalar(1e-16))
    sc.testing.assert_allclose(sc.dot(ez, ex), sc.scalar(0.0), atol=sc.scalar(1e-16))


def test_beam_aligned_unit_vectors_requires_orthogonal_inputs():
    with pytest.raises(
        ValueError, match='`gravity` and `incident_beam` must be orthogonal'
    ):
        beamline.beam_aligned_unit_vectors(
            incident_beam=sc.vector([0.0, 0.0, 3.1], unit='mm'),
            gravity=sc.vector([0.0, -4.6, 1.0], unit='m/s/s'),
        )
