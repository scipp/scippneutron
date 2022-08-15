# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

from hypothesis import assume, given, settings
from hypothesis import strategies as st
import numpy as np
import scipp.constants as const
from scipp.testing import strategies as scst
import scipp as sc

from scippneutron.conversions import tof as tof_conv

# TODO test that precision is preserved


def time_unit():
    return st.sampled_from(('s', 'ms', 'us'))


def space_unit():
    return st.sampled_from(('m', 'cm', 'mm', 'angstrom'))


def energy_unit():
    return st.sampled_from(('meV', 'J'))


def angle_unit():
    return st.sampled_from(('deg', 'rad'))


def element_args():
    return {
        'allow_nan': False,
        'allow_infinity': False,
        'allow_subnormal': False,
        'min_value': 1e-2,
        'max_value': 1e6
    }


def simple_variables(unit=None):
    return scst.variables(ndim=1,
                          with_variances=False,
                          unit=unit,
                          dtype='float',
                          elements=element_args())


def simple_n_variables(n, unit=None):
    return scst.n_variables(n,
                            ndim=1,
                            with_variances=False,
                            unit=unit,
                            dtype='float',
                            elements=element_args())


@given(tof=simple_variables(unit=time_unit()),
       Ltotal=simple_variables(unit=space_unit()))
@settings(max_examples=20)
def test_wavelength_from_tof(tof, Ltotal):
    assume(tof.dims != Ltotal.dims)
    wavelength = tof_conv.wavelength_from_tof(tof=tof, Ltotal=Ltotal)
    assert sc.allclose(wavelength,
                       sc.to_unit(const.h * tof / const.m_n / Ltotal, unit='angstrom'))


@given(tof=simple_variables(unit=time_unit()),
       Ltotal_and_two_theta=simple_n_variables(2, unit=space_unit()),
       two_theta_unit=angle_unit())
@settings(max_examples=20)
def test_dspacing_from_tof(tof, Ltotal_and_two_theta, two_theta_unit):
    Ltotal, two_theta = Ltotal_and_two_theta
    two_theta.unit = two_theta_unit
    assume(tof.dims != Ltotal.dims)
    dspacing = tof_conv.dspacing_from_tof(tof=tof, Ltotal=Ltotal, two_theta=two_theta)
    assert sc.allclose(
        dspacing,
        sc.to_unit(const.h * tof / const.m_n / Ltotal / 2 / sc.sin(two_theta / 2),
                   unit='angstrom'))


@given(tof=simple_variables(unit=time_unit()),
       Ltotal=simple_variables(unit=space_unit()))
@settings(max_examples=20)
def test_energy_from_tof(tof, Ltotal):
    assume(tof.dims != Ltotal.dims)
    energy = tof_conv.energy_from_tof(tof=tof, Ltotal=Ltotal)
    assert sc.allclose(energy, sc.to_unit(const.m_n * Ltotal**2 / 2 / tof**2,
                                          unit='meV'))


def test_energy_transfer_direct_from_tof():
    # TODO
    pass


def test_energy_transfer_indirect_from_tof():
    # TODO
    pass


@given(wavelength=simple_variables(unit=space_unit()))
@settings(max_examples=20)
def test_energy_from_wavelength(wavelength):
    energy = tof_conv.energy_from_wavelength(wavelength=wavelength)
    assert sc.allclose(
        energy, sc.to_unit(const.h**2 / 2 / const.m_n / wavelength**2, unit='meV'))


@given(energy=simple_variables(unit=energy_unit()))
@settings(max_examples=20)
def test_wavelength_from_energy(energy):
    wavelength = tof_conv.wavelength_from_energy(energy=energy)
    assert sc.allclose(
        wavelength,
        sc.to_unit(const.h / sc.sqrt(2 * const.m_n * energy), unit='angstrom'))


@given(wavelength=simple_variables(unit=space_unit()),
       two_theta=simple_variables(unit=angle_unit()))
@settings(max_examples=20)
def test_Q_from_wavelength(wavelength, two_theta):
    assume(wavelength.dims != two_theta.dims)
    Q = tof_conv.Q_from_wavelength(wavelength=wavelength, two_theta=two_theta)
    assert sc.allclose(Q, 4 * np.pi * sc.sin(two_theta / 2) / wavelength)


@given(Q=simple_variables(unit=space_unit()),
       two_theta=simple_variables(unit=angle_unit()))
@settings(max_examples=20)
def test_wavelength_from_Q(Q, two_theta):
    assume(Q.dims != two_theta.dims)
    Q.unit = f'1/{Q.unit}'
    wavelength = tof_conv.wavelength_from_Q(Q=Q, two_theta=two_theta)
    assert sc.allclose(
        wavelength, sc.to_unit(4 * np.pi * sc.sin(two_theta / 2) / Q, unit='angstrom'))


@given(wavelength=simple_variables(unit=space_unit()),
       two_theta=simple_variables(unit=angle_unit()))
@settings(max_examples=20)
def test_dspacing_from_wavelength(wavelength, two_theta):
    assume(wavelength.dims != two_theta.dims)
    dspacing = tof_conv.dspacing_from_wavelength(wavelength=wavelength,
                                                 two_theta=two_theta)
    assert sc.allclose(
        dspacing, sc.to_unit(wavelength / 2 / sc.sin(two_theta / 2), unit='angstrom'))


# TODO more
