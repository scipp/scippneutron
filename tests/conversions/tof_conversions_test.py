# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
import numpy as np
import scipp.constants as const
from scipp.testing import strategies as scst
import scipp as sc

from scippneutron.conversions import tof as tof_conv

# TODO test that precision is preserved


def time_unit():
    return st.sampled_from(('s', 'ms', 'us'))


global_settings = {
    'max_examples': 20,
    # Generating variables needs a lot of memory. Nothing we can fix in scippneutron.
    'suppress_health_check': [HealthCheck.data_too_large]
}


def element_args():
    return {
        'allow_nan': False,
        'allow_infinity': False,
        'allow_subnormal': False,
        'min_value': 1e-2,
        'max_value': 1e6
    }


def simple_variables(dims, unit):
    return scst.variables(sizes=st.dictionaries(keys=dims,
                                                values=st.integers(min_value=1,
                                                                   max_value=4),
                                                min_size=1,
                                                max_size=1),
                          unit=unit,
                          with_variances=False,
                          dtype='float',
                          elements=element_args())


def n_simple_variables(n, dims, unit):
    return scst.n_variables(n,
                            sizes=st.dictionaries(keys=dims,
                                                  values=st.integers(min_value=1,
                                                                     max_value=4),
                                                  min_size=1,
                                                  max_size=1),
                            unit=unit,
                            with_variances=False,
                            dtype='float',
                            elements=element_args())


def time_variables():
    return simple_variables(dims=st.sampled_from(('time', 't', 'tof')),
                            unit=st.sampled_from(('s', 'ms', 'us')))


def space_variables():
    return simple_variables(dims=st.sampled_from(('position', 'spectrum', 'x')),
                            unit=st.sampled_from(('m', 'cm', 'angstrom')))


def n_space_variables(n):
    return n_simple_variables(n,
                              dims=st.sampled_from(('position', 'spectrum', 'x')),
                              unit=st.sampled_from(('m', 'cm', 'angstrom')))


def angle_variables():
    return simple_variables(dims=st.sampled_from(('angle', 'two_theta', 'alpha')),
                            unit=st.sampled_from(('deg', 'rad')))


def energy_variables():
    return simple_variables(dims=st.sampled_from(('energy', 'E', 'energy_transfer')),
                            unit=st.sampled_from(('meV', 'J')))


@given(tof=time_variables(), Ltotal=space_variables())
@settings(**global_settings)
def test_wavelength_from_tof(tof, Ltotal):
    wavelength = tof_conv.wavelength_from_tof(tof=tof, Ltotal=Ltotal)
    assert sc.allclose(wavelength,
                       sc.to_unit(const.h * tof / const.m_n / Ltotal, unit='angstrom'))


@given(tof=time_variables(),
       Ltotal_and_two_theta=n_space_variables(2),
       two_theta_unit=st.sampled_from(('deg', 'rad')))
@settings(**global_settings)
def test_dspacing_from_tof(tof, Ltotal_and_two_theta, two_theta_unit):
    Ltotal, two_theta = Ltotal_and_two_theta
    two_theta.unit = two_theta_unit
    dspacing = tof_conv.dspacing_from_tof(tof=tof, Ltotal=Ltotal, two_theta=two_theta)
    assert sc.allclose(
        dspacing,
        sc.to_unit(const.h * tof / const.m_n / Ltotal / 2 / sc.sin(two_theta / 2),
                   unit='angstrom'))


@given(tof=time_variables(), Ltotal=space_variables())
@settings(**global_settings)
def test_energy_from_tof(tof, Ltotal):
    energy = tof_conv.energy_from_tof(tof=tof, Ltotal=Ltotal)
    assert sc.allclose(energy, sc.to_unit(const.m_n * Ltotal**2 / 2 / tof**2,
                                          unit='meV'))


@given(tof=time_variables(),
       L1_and_L2=n_space_variables(2),
       L2_unit=st.sampled_from(('m', 'mm', 'cm')),
       incident_energy=energy_variables())
@settings(**global_settings)
def test_energy_transfer_direct_from_tof(tof, L1_and_L2, L2_unit, incident_energy):
    L1, L2 = L1_and_L2
    L2.unit = L2_unit
    # Energies are always > 0. This matters for detection of invalid tof values.
    incident_energy = abs(incident_energy) * 1.0001

    energy_transfer = tof_conv.energy_transfer_direct_from_tof(
        tof=tof, L1=L1, L2=L2, incident_energy=incident_energy)

    t0 = sc.to_unit(sc.sqrt(const.m_n * L1**2 / 2 / incident_energy), tof.unit)
    expected = incident_energy - sc.to_unit(const.m_n * L2**2 / 2 /
                                            (tof - t0)**2, incident_energy.unit)
    expected = sc.where(tof >= t0, expected, sc.scalar(np.nan,
                                                       unit=incident_energy.unit))
    assert sc.allclose(energy_transfer, expected, equal_nan=True)


@given(tof=time_variables(),
       L1_and_L2=n_space_variables(2),
       L2_unit=st.sampled_from(('m', 'mm', 'cm')),
       final_energy=energy_variables())
@settings(**global_settings)
def test_energy_transfer_indirect_from_tof(tof, L1_and_L2, L2_unit, final_energy):
    L1, L2 = L1_and_L2
    L2.unit = L2_unit
    # Energies are always > 0. This matters for detection of invalid tof values.
    final_energy = abs(final_energy) * 1.0001

    energy_transfer = tof_conv.energy_transfer_indirect_from_tof(
        tof=tof, L1=L1, L2=L2, final_energy=final_energy)

    t0 = sc.to_unit(sc.sqrt(const.m_n * L2**2 / 2 / final_energy), tof.unit)
    expected = sc.to_unit(const.m_n * L1**2 / 2 /
                          (tof - t0)**2, final_energy.unit) - final_energy
    expected = sc.where(tof >= t0, expected, sc.scalar(np.nan, unit=final_energy.unit))
    assert sc.allclose(energy_transfer, expected, equal_nan=True)


@given(wavelength=space_variables())
@settings(**global_settings)
def test_energy_from_wavelength(wavelength):
    energy = tof_conv.energy_from_wavelength(wavelength=wavelength)
    assert sc.allclose(
        energy, sc.to_unit(const.h**2 / 2 / const.m_n / wavelength**2, unit='meV'))


@given(energy=energy_variables())
@settings(**global_settings)
def test_wavelength_from_energy(energy):
    wavelength = tof_conv.wavelength_from_energy(energy=energy)
    assert sc.allclose(
        wavelength,
        sc.to_unit(const.h / sc.sqrt(2 * const.m_n * energy), unit='angstrom'))


@given(wavelength=space_variables(), two_theta=angle_variables())
@settings(**global_settings)
def test_Q_from_wavelength(wavelength, two_theta):
    Q = tof_conv.Q_from_wavelength(wavelength=wavelength, two_theta=two_theta)
    assert sc.allclose(Q, 4 * np.pi * sc.sin(two_theta / 2) / wavelength)


@given(Q=space_variables(), two_theta=angle_variables())
@settings(**global_settings)
def test_wavelength_from_Q(Q, two_theta):
    Q.unit = f'1/{Q.unit}'
    wavelength = tof_conv.wavelength_from_Q(Q=Q, two_theta=two_theta)
    assert sc.allclose(
        wavelength, sc.to_unit(4 * np.pi * sc.sin(two_theta / 2) / Q, unit='angstrom'))


@given(wavelength=space_variables(), two_theta=angle_variables())
@settings(**global_settings)
def test_dspacing_from_wavelength(wavelength, two_theta):
    dspacing = tof_conv.dspacing_from_wavelength(wavelength=wavelength,
                                                 two_theta=two_theta)
    assert sc.allclose(
        dspacing, sc.to_unit(wavelength / 2 / sc.sin(two_theta / 2), unit='angstrom'))


@given(energy=energy_variables(), two_theta=angle_variables())
@settings(**global_settings)
def test_dspacing_from_energy(energy, two_theta):
    dspacing = tof_conv.dspacing_from_energy(energy=energy, two_theta=two_theta)
    assert sc.allclose(
        dspacing,
        sc.to_unit(const.h / sc.sqrt(8 * const.m_n * energy) / sc.sin(two_theta / 2),
                   unit='angstrom'))
