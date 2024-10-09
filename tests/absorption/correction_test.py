import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_allclose
from scipy.integrate import quad

from scippneutron.absorption import Cylinder, Material, compute_transmission_map
from scippneutron.atoms import ScatteringParams


def transmission_fraction_case1(effective_attenuation_factor):
    '''Transmission fraction at a point on the z-axis
    infinitely far away when the sample is a cylinder
    symmetrical around y, with 1mm radius and 1mm height.
    '''
    return (
        2
        / np.pi
        * quad(
            lambda x: (
                np.sqrt(1 - x**2)
                * np.exp(-2 * effective_attenuation_factor * np.sqrt(1 - x**2))
            ),
            -1,
            1,
        )[0]
    )


@pytest.mark.parametrize('scattering_cross_section', [0.1, 0.5, 1.0])
def test_compute_transmission_map(scattering_cross_section):
    material = Material(
        ScatteringParams(
            'Fake',
            absorption_cross_section=sc.scalar(0.0, unit='mm**2'),
            total_scattering_cross_section=sc.scalar(
                scattering_cross_section, unit='mm**2'
            ),
        ),
        sc.scalar(1.0, unit='1/mm**3'),
    )
    cylinder = Cylinder(
        sc.vector([0, 1.0, 0]),
        sc.vector([0, 0, 0.0], unit='mm'),
        sc.scalar(1.0, unit='mm'),
        sc.scalar(1.0, unit='mm'),
    )

    tm = compute_transmission_map(
        cylinder,
        material,
        beam_direction=sc.vector([0, 0, 1]),
        wavelength=sc.linspace('wavelength', 1.0, 1, 1, unit='angstrom'),
        detector_position=sc.vectors(dims='x', values=[[0, 0, 1]], unit='m'),
        quadrature_kind='expensive',
    )
    assert_allclose(
        tm['wavelength', 0]['x', 0].data,
        sc.scalar(transmission_fraction_case1(scattering_cross_section)),
        rtol=sc.scalar(1e-3),
    )


@pytest.mark.parametrize('scattering_cross_section', [0.1, 0.5, 1.0])
def test_compute_transmission_map_wavelength_dependent(scattering_cross_section):
    material = Material(
        ScatteringParams(
            'Fake',
            # Modify the absorption cross section
            absorption_cross_section=sc.scalar(
                scattering_cross_section / 5, unit='mm**2'
            ),
            total_scattering_cross_section=sc.scalar(0.0, unit='mm**2'),
        ),
        sc.scalar(1.0, unit='1/mm**3'),
    )
    cylinder = Cylinder(
        sc.vector([0, 1.0, 0]),
        sc.vector([0, 0, 0.0], unit='mm'),
        sc.scalar(1.0, unit='mm'),
        sc.scalar(1.0, unit='mm'),
    )

    tm = compute_transmission_map(
        cylinder,
        material,
        beam_direction=sc.vector([0, 0, 1]),
        wavelength=sc.linspace(
            # Set wavelength so that it cancels the
            # modified absorption cross section
            'wavelength',
            5 * 1.7982,
            5 * 1.7982,
            1,
            unit='angstrom',
        ),
        detector_position=sc.vectors(dims='x', values=[[0, 0, 1]], unit='m'),
        quadrature_kind='expensive',
    )
    assert_allclose(
        tm['wavelength', 0]['x', 0].data,
        sc.scalar(transmission_fraction_case1(scattering_cross_section)),
        rtol=sc.scalar(1e-3),
    )
