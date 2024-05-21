# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc

import scippneutron as scn


def test_scattering_params_h():
    params = scn.atoms.ScatteringParams.for_isotope('H')
    expected = scn.atoms.ScatteringParams(
        isotope='H',
        coherent_scattering_length_re=sc.scalar(-3.739, unit='fm'),
        coherent_scattering_length_im=None,
        incoherent_scattering_length_re=None,
        incoherent_scattering_length_im=None,
        coherent_scattering_cross_section=sc.scalar(1.7568, unit='barn'),
        incoherent_scattering_cross_section=sc.scalar(80.26, unit='barn'),
        total_scattering_cross_section=sc.scalar(82.02, unit='barn'),
        absorption_cross_section=sc.scalar(0.3326, unit='barn'),
    )
    assert params == expected


def test_scattering_params_157gd():
    params = scn.atoms.ScatteringParams.for_isotope('157Gd')
    expected = scn.atoms.ScatteringParams(
        isotope='157Gd',
        coherent_scattering_length_re=sc.scalar(-1.14, unit='fm'),
        coherent_scattering_length_im=sc.scalar(-71.9, unit='fm'),
        incoherent_scattering_length_re=sc.scalar(5.0, variance=5.0**2, unit='fm'),
        incoherent_scattering_length_im=sc.scalar(-55.8, unit='fm'),
        coherent_scattering_cross_section=sc.scalar(
            650.0, variance=4.0**2, unit='barn'
        ),
        incoherent_scattering_cross_section=sc.scalar(
            394.0, variance=7.0**2, unit='barn'
        ),
        total_scattering_cross_section=sc.scalar(1044.0, variance=8.0**2, unit='barn'),
        absorption_cross_section=sc.scalar(259000.0, variance=700.0**2, unit='barn'),
    )
    assert params == expected


def test_scattering_params_unknown():
    with pytest.raises(ValueError, match="No entry for element / isotope 'scippium'"):
        scn.atoms.ScatteringParams.for_isotope('scippium')
