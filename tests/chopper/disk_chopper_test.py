# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc
import scipp.constants

from scippneutron.chopper import DiskChopper, DiskChopperType


@pytest.mark.parametrize(
    't',
    (
        ('Chopper type single', DiskChopperType.single),
        ('single', DiskChopperType.single),
        ('contra_rotating_pair', DiskChopperType.contra_rotating_pair),
        ('synchro_pair', DiskChopperType.synchro_pair),
        (DiskChopperType.single, DiskChopperType.single),
        (DiskChopperType.contra_rotating_pair, DiskChopperType.contra_rotating_pair),
        (DiskChopperType.synchro_pair, DiskChopperType.synchro_pair),
    ),
)
def test_chopper_type_init(t):
    arg, expected = t
    ch = DiskChopper(
        typ=arg,
        rotation_speed=sc.scalar(1.0, unit='Hz'),
        position=sc.vector([0, 0, 0], unit='m'),
    )
    assert isinstance(ch.typ, DiskChopperType)
    assert ch.typ == expected


def test_bad_chopper_type_init():
    with pytest.raises(ValueError):
        DiskChopper(
            typ="contra",
            rotation_speed=sc.scalar(1.0, unit='Hz'),
            position=sc.vector([0, 0, 0], unit='m'),
        )


def test_rotation_speed_must_be_frequency():
    with pytest.raises(sc.UnitError):
        DiskChopper(
            typ="single",
            rotation_speed=sc.scalar(1.0, unit='m/s'),
            position=sc.vector([0, 0, 0], unit='m'),
        )


def test_eq():
    ch1 = DiskChopper(
        typ='single',
        rotation_speed=sc.scalar(14.0, unit='Hz'),
        position=sc.vector([0, 0, 0], unit='m'),
        name="ch",
    )
    ch2 = DiskChopper(
        typ='single',
        rotation_speed=sc.scalar(14.0, unit='Hz'),
        position=sc.vector([0, 0, 0], unit='m'),
        name="ch",
    )
    assert ch1 == ch2


@pytest.mark.parametrize(
    'replacement',
    (
        ('typ', 'contra_rotating_pair'),
        ('rotation_speed', sc.scalar(13.0, unit='Hz')),
        ('position', sc.vector([1, 0, 0], unit='m')),
        ('name', 'ch2'),
        ('radius', sc.scalar(0.5, unit='m')),
        ('phase', sc.scalar(15, unit='deg')),
        ('slits', 5),
        ('slit_height', sc.scalar(0.14, unit='cm')),
        ('slit_edges', sc.array(dims=['edge'], values=[0.1, 0.3], unit='rad')),
    ),
)
def test_neq(replacement):
    args = dict(
        typ='single',
        rotation_speed=sc.scalar(14.0, unit='Hz'),
        position=sc.vector([0, 0, 0], unit='m'),
        name="ch",
    )
    ch1 = DiskChopper(**args)
    ch2 = DiskChopper(**{**args, replacement[0]: replacement[1]})
    assert ch1 != ch2


def test_slit_begin_end_no_slit():
    ch = DiskChopper(
        typ='single',
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(5.12, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[], unit='deg'),
    )
    assert sc.identical(ch.slit_begin, sc.array(dims=['slit'], values=[], unit='deg'))
    assert sc.identical(ch.slit_end, sc.array(dims=['slit'], values=[], unit='deg'))


def test_slit_begin_end_one_slit():
    ch = DiskChopper(
        typ='single',
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(5.12, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[13, 43], unit='deg'),
    )
    assert sc.identical(ch.slit_begin, sc.array(dims=['slit'], values=[13], unit='deg'))
    assert sc.identical(ch.slit_end, sc.array(dims=['slit'], values=[43], unit='deg'))


def test_slit_begin_end_two_slits():
    ch = DiskChopper(
        typ='single',
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(5.12, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[0, 60, 124, 126], unit='deg'),
    )
    assert sc.identical(
        ch.slit_begin, sc.array(dims=['slit'], values=[0, 124], unit='deg')
    )
    assert sc.identical(
        ch.slit_end, sc.array(dims=['slit'], values=[60, 126], unit='deg')
    )


def test_slit_begin_end_across_0():
    ch = DiskChopper(
        typ='single',
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(5.12, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[340.0, 382.0], unit='deg'),
    )
    assert sc.identical(
        ch.slit_begin, sc.array(dims=['slit'], values=[340.0], unit='deg')
    )
    assert sc.identical(
        ch.slit_end, sc.array(dims=['slit'], values=[382.0], unit='deg')
    )


# TODO negative rotation speed
def test_time_open_close_no_slit():
    ch = DiskChopper(
        typ='single',
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(5.12, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[], unit='deg'),
    )
    assert sc.identical(ch.time_open(), sc.array(dims=['slit'], values=[], unit='s'))
    assert sc.identical(ch.time_close(), sc.array(dims=['slit'], values=[], unit='s'))
    assert sc.identical(
        ch.open_duration(), sc.array(dims=['slit'], values=[], unit='s')
    )


def test_time_open_close_only_slit():
    ch = DiskChopper(
        typ='single',
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(5.12, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[0.0, 360.0], unit='deg'),
    )
    factor = (
        sc.constants.pi.value
        / 180  # to rad
        / (2 * sc.constants.pi.value)  # to angular frequency
        / 5.12
    )  # to time based on rotation speed
    assert sc.identical(ch.time_open(), sc.array(dims=['slit'], values=[0.0], unit='s'))
    assert sc.allclose(
        ch.time_close(), sc.array(dims=['slit'], values=[360.0 * factor], unit='s')
    )
    assert sc.allclose(
        ch.open_duration(), sc.array(dims=['slit'], values=[1 / 5.12], unit='s')
    )


def test_time_open_close_two_slits():
    ch = DiskChopper(
        typ='single',
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(3.29, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[0.5, 0.8, 2.4, 2.5], unit='rad'),
    )
    factor = 1 / 3.29 / (2 * sc.constants.pi.value)
    assert sc.allclose(
        ch.time_open(),
        sc.array(dims=['slit'], values=[0.5 * factor, 2.4 * factor], unit='s'),
    )
    assert sc.allclose(
        ch.time_close(),
        sc.array(dims=['slit'], values=[0.8 * factor, 2.5 * factor], unit='s'),
    )
    assert sc.allclose(
        ch.open_duration(),
        sc.array(dims=['slit'], values=[0.3 * factor, 0.1 * factor], unit='s'),
    )


def test_time_open_close_slit_across_0():
    ch = DiskChopper(
        typ='single',
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(1.2, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[355.0, 372.0], unit='deg'),
    )
    factor = (
        sc.constants.pi.value
        / 180  # to rad
        / (2 * sc.constants.pi.value)  # to angular frequency
        / 1.2
    )  # to time based on rotation speed
    assert sc.allclose(
        ch.time_open(), sc.array(dims=['slit'], values=[355.0 * factor], unit='s')
    )
    # Like the edge, the time does not wrap around and the
    # closing time is > rotation period.
    assert sc.allclose(
        ch.time_close(), sc.array(dims=['slit'], values=[372.0 * factor], unit='s')
    )
    assert sc.allclose(
        ch.open_duration(), sc.array(dims=['slit'], values=[17.0 * factor], unit='s')
    )


def test_time_open_close_no_slit_with_phase():
    ch = DiskChopper(
        typ='single',
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(5.12, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[], unit='deg'),
        phase=sc.scalar(3.51, unit='rad'),
    )
    assert sc.identical(ch.time_open(), sc.array(dims=['slit'], values=[], unit='s'))
    assert sc.identical(ch.time_close(), sc.array(dims=['slit'], values=[], unit='s'))
    assert sc.identical(
        ch.open_duration(), sc.array(dims=['slit'], values=[], unit='s')
    )


# TODO negative rotation speed
@pytest.mark.parametrize('phase', (0.0, 5.2, -1.4))
@pytest.mark.parametrize('phase_unit', ('deg', 'rad'))
def test_time_open_close_only_slit_with_phase(phase, phase_unit):
    ch = DiskChopper(
        typ='single',
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(5.12, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[0.0, 360.0], unit='deg'),
        phase=sc.scalar(phase, unit=phase_unit),
    )
    phase_deg = sc.scalar(phase, unit=phase_unit).to(unit='deg').value
    factor = (
        sc.constants.pi.value
        / 180  # to rad
        / (2 * sc.constants.pi.value)  # to angular frequency
        / 5.12
    )  # to time based on rotation speed
    assert sc.allclose(
        ch.time_open(),
        sc.array(dims=['slit'], values=[(0.0 + phase_deg) * factor], unit='s'),
    )
    assert sc.allclose(
        ch.time_close(),
        sc.array(dims=['slit'], values=[(360.0 + phase_deg) * factor], unit='s'),
    )
    assert sc.allclose(
        ch.open_duration(), sc.array(dims=['slit'], values=[1 / 5.12], unit='s')
    )


@pytest.mark.parametrize('phase', (0.0, 1.3, 5.4, -2.2))
@pytest.mark.parametrize('phase_unit', ('deg', 'rad'))
def test_open_duration_does_not_depend_on_phase(phase, phase_unit):
    ch = DiskChopper(
        typ='single',
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(14.1, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[0.1, 0.6, 1.9, 2.3], unit='rad'),
        phase=sc.scalar(phase, unit=phase_unit),
    )
    factor = 1 / 14.1 / (2 * sc.constants.pi.value)
    assert sc.allclose(
        ch.open_duration(),
        sc.array(dims=['slit'], values=[0.5 * factor, 0.4 * factor], unit='s'),
    )


# TODO negative rotation_speed
# TODO check rotation direction for times
