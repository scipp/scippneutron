# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
import numpy as np
import pytest
import scipp.constants as const
from scipp.testing import strategies as scst
import scipp as sc

from scippneutron.conversion import tof as tof_conv

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


@pytest.mark.parametrize('tof_dtype', ('float64', 'int64', 'int32'))
@pytest.mark.parametrize('Ltotal_dtype', ('float64', 'float32', 'int64', 'int32'))
def test_wavelength_from_tof_double_precision(tof_dtype, Ltotal_dtype):
    tof = sc.scalar(1.2, unit='s', dtype=tof_dtype)
    Ltotal = sc.scalar(10.1, unit='m', dtype=Ltotal_dtype)
    assert tof_conv.wavelength_from_tof(tof=tof, Ltotal=Ltotal).dtype == 'float64'


@pytest.mark.parametrize('Ltotal_dtype', ('float64', 'float32', 'int64', 'int32'))
def test_wavelength_from_tof_single_precision(Ltotal_dtype):
    tof = sc.scalar(1.2, unit='s', dtype='float32')
    Ltotal = sc.scalar(10.1, unit='m', dtype=Ltotal_dtype)
    assert tof_conv.wavelength_from_tof(tof=tof, Ltotal=Ltotal).dtype == 'float32'


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


@pytest.mark.parametrize('tof_dtype', ('float64', 'int64', 'int32'))
@pytest.mark.parametrize('Ltotal_dtype', ('float64', 'float32', 'int64', 'int32'))
@pytest.mark.parametrize('two_theta_dtype', ('float64', 'float32', 'int64', 'int32'))
def test_dspacing_from_tof_double_precision(tof_dtype, Ltotal_dtype, two_theta_dtype):
    tof = sc.scalar(52.0, unit='s', dtype=tof_dtype)
    Ltotal = sc.scalar(0.341, unit='m', dtype=Ltotal_dtype)
    two_theta = sc.scalar(1.68, unit='rad', dtype=two_theta_dtype)
    assert tof_conv.dspacing_from_tof(tof=tof, Ltotal=Ltotal,
                                      two_theta=two_theta).dtype == 'float64'


@pytest.mark.parametrize('Ltotal_dtype', ('float64', 'float32', 'int64', 'int32'))
@pytest.mark.parametrize('two_theta_dtype', ('float64', 'float32', 'int64', 'int32'))
def test_dspacing_from_tof_single_precision(Ltotal_dtype, two_theta_dtype):
    tof = sc.scalar(52.0, unit='s', dtype='float32')
    Ltotal = sc.scalar(0.341, unit='m', dtype=Ltotal_dtype)
    two_theta = sc.scalar(1.68, unit='rad', dtype=two_theta_dtype)
    assert tof_conv.dspacing_from_tof(tof=tof, Ltotal=Ltotal,
                                      two_theta=two_theta).dtype == 'float32'


@given(tof=time_variables(), Ltotal=space_variables())
@settings(**global_settings)
def test_energy_from_tof(tof, Ltotal):
    energy = tof_conv.energy_from_tof(tof=tof, Ltotal=Ltotal)
    assert sc.allclose(energy, sc.to_unit(const.m_n * Ltotal**2 / 2 / tof**2,
                                          unit='meV'))


@pytest.mark.parametrize('tof_dtype', ('float64', 'int64'))
@pytest.mark.parametrize('Ltotal_dtype', ('float64', 'float32', 'int64'))
def test_energy_from_tof_double_precision(tof_dtype, Ltotal_dtype):
    tof = sc.scalar(478.9, unit='s', dtype=tof_dtype)
    Ltotal = sc.scalar(1.256, unit='m', dtype=Ltotal_dtype)
    assert tof_conv.energy_from_tof(tof=tof, Ltotal=Ltotal).dtype == 'float64'


@pytest.mark.parametrize('Ltotal_dtype', ('float64', 'float32', 'int64'))
def test_energy_from_tof_single_precision(Ltotal_dtype):
    tof = sc.scalar(478.9, unit='s', dtype='float32')
    Ltotal = sc.scalar(1.256, unit='m', dtype=Ltotal_dtype)
    assert tof_conv.energy_from_tof(tof=tof, Ltotal=Ltotal).dtype == 'float32'


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


@pytest.mark.parametrize('wavelength_dtype', ('float64', 'int64'))
def test_energy_from_wavelength_double_precision(wavelength_dtype):
    wavelength = sc.scalar(60.5, unit='m', dtype=wavelength_dtype)
    assert tof_conv.energy_from_wavelength(wavelength=wavelength).dtype == 'float64'


def test_energy_from_wavelength_single_precision():
    wavelength = sc.scalar(60.5, unit='m', dtype='float32')
    assert tof_conv.energy_from_wavelength(wavelength=wavelength).dtype == 'float32'


@given(energy=energy_variables())
@settings(**global_settings)
def test_wavelength_from_energy(energy):
    wavelength = tof_conv.wavelength_from_energy(energy=energy)
    assert sc.allclose(
        wavelength,
        sc.to_unit(const.h / sc.sqrt(2 * const.m_n * energy), unit='angstrom'))


@pytest.mark.parametrize('energy_dtype', ('float64', 'int64'))
def test_wavelength_from_energy_double_precision(energy_dtype):
    energy = sc.scalar(61.0, unit='meV', dtype=energy_dtype)
    assert tof_conv.wavelength_from_energy(energy=energy).dtype == 'float64'


def test_wavelength_from_energy_single_precision():
    energy = sc.scalar(61.0, unit='meV', dtype='float32')
    assert tof_conv.wavelength_from_energy(energy=energy).dtype == 'float32'


@given(wavelength=space_variables(), two_theta=angle_variables())
@settings(**global_settings)
def test_Q_from_wavelength(wavelength, two_theta):
    Q = tof_conv.Q_from_wavelength(wavelength=wavelength, two_theta=two_theta)
    assert sc.allclose(Q, 4 * np.pi * sc.sin(two_theta / 2) / wavelength)


@pytest.mark.parametrize('wavelength_dtype', ('float64', 'int64'))
@pytest.mark.parametrize('two_theta_dtype', ('float64', 'float32', 'int64'))
def test_Q_from_wavelength_double_precision(wavelength_dtype, two_theta_dtype):
    wavelength = sc.scalar(3.51, unit='s', dtype=wavelength_dtype)
    two_theta = sc.scalar(0.041, unit='deg', dtype=two_theta_dtype)
    assert tof_conv.Q_from_wavelength(wavelength=wavelength,
                                      two_theta=two_theta).dtype == 'float64'


@pytest.mark.parametrize('two_theta_dtype', ('float64', 'float32', 'int64'))
def test_Q_from_wavelength_single_precision(two_theta_dtype):
    wavelength = sc.scalar(3.51, unit='s', dtype='float32')
    two_theta = sc.scalar(0.041, unit='deg', dtype=two_theta_dtype)
    assert tof_conv.Q_from_wavelength(wavelength=wavelength,
                                      two_theta=two_theta).dtype == 'float32'


@given(Q=space_variables(), two_theta=angle_variables())
@settings(**global_settings)
def test_wavelength_from_Q(Q, two_theta):
    Q.unit = f'1/{Q.unit}'
    wavelength = tof_conv.wavelength_from_Q(Q=Q, two_theta=two_theta)
    assert sc.allclose(
        wavelength, sc.to_unit(4 * np.pi * sc.sin(two_theta / 2) / Q, unit='angstrom'))


@pytest.mark.parametrize('Q_dtype', ('float64', 'int64'))
@pytest.mark.parametrize('two_theta_dtype', ('float64', 'float32', 'int64'))
def test_wavelength_from_Q_double_precision(Q_dtype, two_theta_dtype):
    Q = sc.scalar(4.151, unit='1/nm', dtype=Q_dtype)
    two_theta = sc.scalar(5.71, unit='deg', dtype=two_theta_dtype)
    assert tof_conv.wavelength_from_Q(Q=Q, two_theta=two_theta).dtype == 'float64'


@pytest.mark.parametrize('two_theta_dtype', ('float64', 'float32', 'int64'))
def test_wavelength_from_Q_single_precision(two_theta_dtype):
    Q = sc.scalar(4.151, unit='1/nm', dtype='float32')
    two_theta = sc.scalar(5.71, unit='deg', dtype=two_theta_dtype)
    assert tof_conv.wavelength_from_Q(Q=Q, two_theta=two_theta).dtype == 'float32'


@given(wavelength=space_variables(), two_theta=angle_variables())
@settings(**global_settings)
def test_dspacing_from_wavelength(wavelength, two_theta):
    dspacing = tof_conv.dspacing_from_wavelength(wavelength=wavelength,
                                                 two_theta=two_theta)
    assert sc.allclose(
        dspacing, sc.to_unit(wavelength / 2 / sc.sin(two_theta / 2), unit='angstrom'))


@pytest.mark.parametrize('wavelength_dtype', ('float64', 'int64'))
@pytest.mark.parametrize('two_theta_dtype', ('float64', 'float32', 'int64'))
def test_dspacing_from_wavelength_double_precision(wavelength_dtype, two_theta_dtype):
    wavelength = sc.scalar(41.4, unit='m', dtype=wavelength_dtype)
    two_theta = sc.scalar(8.4, unit='rad', dtype=two_theta_dtype)
    assert tof_conv.dspacing_from_wavelength(wavelength=wavelength,
                                             two_theta=two_theta).dtype == 'float64'


@pytest.mark.parametrize('two_theta_dtype', ('float64', 'float32', 'int64'))
def test_dspacing_from_wavelength_single_precision(two_theta_dtype):
    wavelength = sc.scalar(41.4, unit='m', dtype='float32')
    two_theta = sc.scalar(8.4, unit='rad', dtype=two_theta_dtype)
    assert tof_conv.dspacing_from_wavelength(wavelength=wavelength,
                                             two_theta=two_theta).dtype == 'float32'


@given(energy=energy_variables(), two_theta=angle_variables())
@settings(**global_settings)
def test_dspacing_from_energy(energy, two_theta):
    dspacing = tof_conv.dspacing_from_energy(energy=energy, two_theta=two_theta)
    assert sc.allclose(
        dspacing,
        sc.to_unit(const.h / sc.sqrt(8 * const.m_n * energy) / sc.sin(two_theta / 2),
                   unit='angstrom'))


@pytest.mark.parametrize('energy_dtype', ('float64', 'int64'))
@pytest.mark.parametrize('two_theta_dtype', ('float64', 'float32', 'int64'))
def test_dspacing_from_energy_double_precision(energy_dtype, two_theta_dtype):
    energy = sc.scalar(26.90, unit='J', dtype=energy_dtype)
    two_theta = sc.scalar(1.985, unit='rad', dtype=two_theta_dtype)
    assert tof_conv.dspacing_from_energy(energy=energy,
                                         two_theta=two_theta).dtype == 'float64'


@pytest.mark.parametrize('two_theta_dtype', ('float64', 'float32', 'int64'))
def test_dspacing_from_energy_single_precision(two_theta_dtype):
    energy = sc.scalar(26.90, unit='J', dtype='float32')
    two_theta = sc.scalar(1.985, unit='rad', dtype=two_theta_dtype)
    assert tof_conv.dspacing_from_energy(energy=energy,
                                         two_theta=two_theta).dtype == 'float32'
