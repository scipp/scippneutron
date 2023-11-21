# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import scipp as sc

from scippneutron.tof import chopper_cascade


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
