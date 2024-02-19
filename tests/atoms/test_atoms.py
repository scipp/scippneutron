# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc
from scipp.testing import assert_identical

import scippneutron as scn


def test_scattering_params_h():
    params = scn.atoms.scattering_params('H')
    assert len(params) == 8
    assert_identical(
        params['bound_coherent_scattering_length_re'], sc.scalar(-3.739, unit='fm')
    )
    assert params['bound_coherent_scattering_length_im'] is None
    assert params['bound_incoherent_scattering_length_re'] is None
    assert params['bound_incoherent_scattering_length_im'] is None
    assert_identical(
        params['bound_coherent_scattering_cross_section'],
        sc.scalar(1.7568, unit='barn'),
    )
    assert_identical(
        params['bound_incoherent_scattering_cross_section'],
        sc.scalar(80.26, unit='barn'),
    )
    assert_identical(
        params['total_bound_scattering_cross_section'], sc.scalar(82.02, unit='barn')
    )
    assert_identical(params['absorption_cross_section'], sc.scalar(0.3326, unit='barn'))


def test_scattering_params_157gd():
    params = scn.atoms.scattering_params('157Gd')
    assert len(params) == 8
    assert_identical(
        params['bound_coherent_scattering_length_re'], sc.scalar(-1.14, unit='fm')
    )
    assert_identical(
        params['bound_coherent_scattering_length_im'], sc.scalar(-71.9, unit='fm')
    )
    assert_identical(
        params['bound_incoherent_scattering_length_re'],
        sc.scalar(5.0, variance=5.0**2, unit='fm'),
    )
    assert_identical(
        params['bound_incoherent_scattering_length_im'], sc.scalar(-55.8, unit='fm')
    )
    assert_identical(
        params['bound_coherent_scattering_cross_section'],
        sc.scalar(650.0, variance=4.0**2, unit='barn'),
    )
    assert_identical(
        params['bound_incoherent_scattering_cross_section'],
        sc.scalar(394.0, variance=7.0**2, unit='barn'),
    )
    assert_identical(
        params['total_bound_scattering_cross_section'],
        sc.scalar(1044.0, variance=8.0**2, unit='barn'),
    )
    assert_identical(
        params['absorption_cross_section'],
        sc.scalar(259000.0, variance=700.0**2, unit='barn'),
    )


def test_scattering_params_unknown():
    with pytest.raises(ValueError):
        scn.atoms.scattering_params('scippium')
