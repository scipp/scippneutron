# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import pytest
import scipp as sc
from scipp.testing import assert_identical

from scippneutron.tof import chopper_cascade


def test_subframe_init_raises_if_time_and_wavelength_have_different_dims() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 1.0, 1.0, 0.0], unit='s')
    wavelength = sc.array(dims=['vertex'], values=[1.0, 1.0, 2.0, 2.0], unit='angstrom')
    with pytest.raises(sc.DimensionError):
        chopper_cascade.Subframe(time=time, wavelength=wavelength[0:3])
    with pytest.raises(sc.DimensionError):
        chopper_cascade.Subframe(
            time=time, wavelength=wavelength.rename_dims({'vertex': 'point'})
        )


def test_subframe_init_raises_if_time_cannot_be_converted_to_seconds() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 1.0, 1.0, 0.0], unit='m')
    wavelength = sc.array(dims=['vertex'], values=[1.0, 1.0, 2.0, 2.0], unit='angstrom')
    with pytest.raises(sc.UnitError):
        chopper_cascade.Subframe(time=time, wavelength=wavelength)


def test_subframe_init_raises_if_wavelength_cannot_be_converted_to_angstrom() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 1.0, 1.0, 0.0], unit='s')
    wavelength = sc.array(dims=['vertex'], values=[1.0, 1.0, 2.0, 2.0], unit='s')
    with pytest.raises(sc.UnitError):
        chopper_cascade.Subframe(time=time, wavelength=wavelength)


def test_subframe_is_regular() -> None:
    # Triangle with last point after base
    time = sc.array(dims=['vertex'], values=[0.0, 2.0, 3.0], unit='s')
    wavelength = sc.array(dims=['vertex'], values=[1.0, 1.0, 2.0], unit='angstrom')
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    assert subframe.is_regular()
    # Triangle with last point inside base
    time = sc.array(dims=['vertex'], values=[0.0, 2.0, 1.0], unit='s')
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    assert not subframe.is_regular()
    # Triangle standing on its tip, to also test the min-wavelength
    time = sc.array(dims=['vertex'], values=[1.0, 0.0, 3.0], unit='s')
    wavelength = sc.array(dims=['vertex'], values=[2.0, 1.0, 2.0], unit='angstrom')
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    assert subframe.is_regular()
    time = sc.array(dims=['vertex'], values=[1.0, 2.0, 3.0], unit='s')
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    assert not subframe.is_regular()


def test_subframe_propagate_by() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 1.0, 1.0, 0.0], unit='s')
    wavelength = sc.array(dims=['vertex'], values=[1.0, 1.0, 2.0, 2.0], unit='angstrom')
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    propagated = subframe.propagate_by(sc.scalar(1.0, unit='m'))
    assert_identical(propagated.wavelength, subframe.wavelength)
    assert (propagated.time > subframe.time).all()
    # Started at same time, but different wavelength
    assert propagated.time[2] > propagated.time[1]
    # Difference should be proportional to wavelength
    dt = propagated.time - subframe.time
    rtol = sc.scalar(1e-12, unit='')
    assert sc.isclose(dt[0], dt[1], atol=sc.scalar(0.0, unit='s'), rtol=rtol)
    assert sc.isclose(dt[2], dt[3], atol=sc.scalar(0.0, unit='s'), rtol=rtol)
    assert sc.allclose(
        (dt[2:3] / dt[1]), sc.scalar(2.0), atol=sc.scalar(0.0), rtol=rtol
    )


def test_subframe_time_is_converted_to_seconds() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 1.0, 3.0, 2.0], unit='ms')
    wavelength = sc.array(dims=['vertex'], values=[1.0, 1.1, 2.1, 2.0], unit='nm')
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    assert_identical(
        subframe.time,
        sc.array(dims=['vertex'], values=[0.0, 0.001, 0.003, 0.002], unit='s'),
    )


def test_subframe_wavelength_is_converted_to_angstrom() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 1.0, 3.0, 2.0], unit='ms')
    wavelength = sc.array(dims=['vertex'], values=[1.0, 1.1, 2.1, 2.0], unit='nm')
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    assert_identical(
        subframe.wavelength,
        sc.array(dims=['vertex'], values=[10.0, 11.0, 21.0, 20.0], unit='angstrom'),
    )


def test_subframe_start_end_properties() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 1.0, 3.0, 2.0], unit='ms')
    wavelength = sc.array(dims=['vertex'], values=[1.0, 1.1, 2.1, 2.0], unit='nm')
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    assert_identical(subframe.start_time, sc.scalar(0.0, unit='s'))
    assert_identical(subframe.end_time, sc.scalar(0.003, unit='s'))
    assert_identical(subframe.start_wavelength, sc.scalar(10.0, unit='angstrom'))
    assert_identical(subframe.end_wavelength, sc.scalar(21.0, unit='angstrom'))
