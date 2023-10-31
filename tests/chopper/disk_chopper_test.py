# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc
import scipp.constants

from scippneutron.chopper import DiskChopper, DiskChopperType


@pytest.mark.parametrize(
    'typ',
    (
        'contra_rotating_pair',
        'synchro_pair',
        DiskChopperType.contra_rotating_pair,
        DiskChopperType.synchro_pair,
    ),
)
def test_chopper_supports_only_single(typ):
    with pytest.raises(NotImplementedError):
        DiskChopper(
            typ=typ,
            rotation_speed=sc.scalar(1.0, unit='Hz'),
            position=sc.vector([0, 0, 0], unit='m'),
        )


def test_rotation_speed_must_be_frequency():
    with pytest.raises(sc.UnitError):
        DiskChopper(
            rotation_speed=sc.scalar(1.0, unit='m/s'),
            position=sc.vector([0, 0, 0], unit='m'),
        )


def test_eq():
    ch1 = DiskChopper(
        rotation_speed=sc.DataArray(
            sc.array(dims=['time'], values=[1, 2], unit='Hz'),
            coords={'time': sc.arange('time', 2, unit='ms')},
        ),
        position=sc.vector([0, 0, 0], unit='m'),
    )
    ch2 = DiskChopper(
        rotation_speed=sc.DataArray(
            sc.array(dims=['time'], values=[1, 2], unit='Hz'),
            coords={'time': sc.arange('time', 2, unit='ms')},
        ),
        position=sc.vector([0, 0, 0], unit='m'),
    )
    assert ch1 == ch2


@pytest.mark.parametrize(
    'replacement',
    (
        ('rotation_speed', sc.scalar(13.0, unit='Hz')),
        ('position', sc.vector([1, 0, 0], unit='m')),
        ('radius', sc.scalar(0.5, unit='m')),
        ('phase', sc.scalar(15, unit='deg')),
        ('slits', 5),
        ('slit_height', sc.scalar(0.14, unit='cm')),
        ('slit_edges', sc.array(dims=['edge'], values=[0.1, 0.3], unit='rad')),
        ('top_dead_center', sc.datetimes(dims=['time'], values=[2, 5, 8], unit='ns')),
    ),
)
def test_neq(replacement):
    args = dict(
        rotation_speed=sc.scalar(14.0, unit='Hz'),
        position=sc.vector([0, 0, 0], unit='m'),
    )
    ch1 = DiskChopper(**args)
    ch2 = DiskChopper(**{**args, replacement[0]: replacement[1]})
    assert ch1 != ch2


def test_slit_edges_converted_from_1d():
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(5.12, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[0, 60, 124, 126], unit='deg'),
    )
    assert sc.identical(
        ch.slit_edges,
        sc.array(dims=['slit', 'edge'], values=[[0, 60], [124, 126]], unit='deg'),
    )


def test_slit_edges_can_be_2d():
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(5.12, unit='Hz'),
        slit_edges=sc.array(
            dims=['slit', 'edge'], values=[[0, 60], [124, 126]], unit='deg'
        ),
    )
    assert sc.identical(
        ch.slit_edges,
        sc.array(dims=['slit', 'edge'], values=[[0, 60], [124, 126]], unit='deg'),
    )


def test_slit_edges_disallowed_ndim():
    with pytest.raises(sc.DimensionError):
        DiskChopper(
            position=sc.vector([0, 0, 0], unit='m'),
            rotation_speed=sc.scalar(5.12, unit='Hz'),
            slit_edges=sc.scalar(30.0, unit='deg'),
        )
    with pytest.raises(sc.DimensionError):
        DiskChopper(
            position=sc.vector([0, 0, 0], unit='m'),
            rotation_speed=sc.scalar(5.12, unit='Hz'),
            slit_edges=sc.array(
                dims=['time', 'slit', 'edge'],
                values=[[[0, 60], [124, 126]]],
                unit='deg',
            ),
        )


def test_2d_slit_edges_must_have_length_2():
    with pytest.raises(sc.DimensionError):
        DiskChopper(
            position=sc.vector([0, 0, 0], unit='m'),
            rotation_speed=sc.scalar(5.12, unit='Hz'),
            slit_edges=sc.array(
                dims=['time', 'slit', 'edge'],
                values=[[0, 60, 90], [124, 126, 270]],
                unit='deg',
            ),
        )


@pytest.mark.parametrize('rotation_speed', (1.0, -1.0))
def slit_edges_must_be_ascending_per_slit(rotation_speed):
    with pytest.raises(ValueError):
        DiskChopper(
            rotation_speed=sc.scalar(rotation_speed, unit='Hz'),
            position=sc.vector([0, 0, 0], unit='m'),
            slit_edges=sc.array(dims=['edge'], values=[0.5, 0.3], unit='rad'),
        )
    with pytest.raises(ValueError):
        DiskChopper(
            rotation_speed=sc.scalar(rotation_speed, unit='Hz'),
            position=sc.vector([0, 0, 0], unit='m'),
            slit_edges=sc.array(dims=['edge'], values=[0.8, 0.6, 2.5, 2.8], unit='rad'),
        )


def test_slit_begin_end_no_slit():
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(5.12, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[], unit='deg'),
    )
    assert sc.identical(ch.slit_begin, sc.array(dims=['slit'], values=[], unit='deg'))
    assert sc.identical(ch.slit_end, sc.array(dims=['slit'], values=[], unit='deg'))


def test_slit_begin_end_one_slit():
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(5.12, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[13, 43], unit='deg'),
    )
    assert sc.identical(ch.slit_begin, sc.array(dims=['slit'], values=[13], unit='deg'))
    assert sc.identical(ch.slit_end, sc.array(dims=['slit'], values=[43], unit='deg'))


def test_slit_begin_end_two_slits():
    ch = DiskChopper(
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


@pytest.mark.parametrize('rotation_speed', (1.0, -1.0))
def test_slit_begin_end_two_slits_unordered(rotation_speed):
    ch = DiskChopper(
        rotation_speed=sc.scalar(rotation_speed, unit='Hz'),
        position=sc.vector([0, 0, 0], unit='m'),
        slit_edges=sc.array(dims=['slit'], values=[2.5, 2.8, 0.8, 1.3], unit='rad'),
    )
    assert sc.identical(
        ch.slit_begin, sc.array(dims=['slit'], values=[2.5, 0.8], unit='rad')
    )
    assert sc.identical(
        ch.slit_end, sc.array(dims=['slit'], values=[2.8, 1.3], unit='rad')
    )


def test_slit_begin_end_across_0():
    ch = DiskChopper(
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


@pytest.mark.parametrize('rotation_speed', (5.12, -3.6))
def test_relative_time_open_close_no_slit(rotation_speed):
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(rotation_speed, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[], unit='deg'),
    )
    assert sc.identical(
        ch.relative_time_open(), sc.array(dims=['slit'], values=[], unit='s')
    )
    assert sc.identical(
        ch.relative_time_close(), sc.array(dims=['slit'], values=[], unit='s')
    )
    assert sc.identical(
        ch.open_duration(), sc.array(dims=['slit'], values=[], unit='s')
    )


@pytest.mark.parametrize('rotation_speed', (5.12, -3.6))
def test_relative_time_open_close_only_slit(rotation_speed):
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(rotation_speed, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[0.0, 360.0], unit='deg'),
    )
    factor = (
        sc.constants.pi.value
        / 180  # to rad
        / (2 * sc.constants.pi.value)  # to angular frequency
        / abs(rotation_speed)
    )  # to time based on rotation speed
    assert sc.allclose(
        ch.relative_time_open(), sc.array(dims=['slit'], values=[0.0], unit='s')
    )
    assert sc.allclose(
        ch.relative_time_close(),
        sc.array(dims=['slit'], values=[360.0 * factor], unit='s'),
    )
    assert sc.allclose(
        ch.open_duration(),
        sc.array(dims=['slit'], values=[1 / abs(rotation_speed)], unit='s'),
    )


@pytest.mark.parametrize(
    'phase',
    (
        sc.scalar(0.0, unit='rad'),
        sc.scalar(1.2, unit='rad'),
        sc.scalar(-50.0, unit='deg'),
    ),
)
def test_relative_time_open_close_single_slit_clockwise(phase):
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(-7.21, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[90.0, 180.0], unit='deg'),
        phase=phase,
    )
    factor = (
        sc.constants.pi.value
        / 180  # to rad
        / (2 * sc.constants.pi.value)  # to angular frequency
        / abs(-7.21)
    )  # to time based on rotation speed
    assert sc.allclose(
        ch.relative_time_open(), sc.array(dims=['slit'], values=[90 * factor], unit='s')
    )
    assert sc.allclose(
        ch.relative_time_close(),
        sc.array(dims=['slit'], values=[180 * factor], unit='s'),
    )
    assert sc.allclose(
        ch.open_duration(), sc.array(dims=['slit'], values=[90 * factor], unit='s')
    )


@pytest.mark.parametrize(
    'phase',
    (
        sc.scalar(0.0, unit='rad'),
        sc.scalar(1.2, unit='rad'),
        sc.scalar(-50.0, unit='deg'),
    ),
)
def test_relative_time_open_close_single_slit_anticlockwise(phase):
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(7.21, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[90.0, 180.0], unit='deg'),
        phase=phase,
    )
    factor = (
        sc.constants.pi.value
        / 180  # to rad
        / (2 * sc.constants.pi.value)  # to angular frequency
        / abs(7.21)
    )  # to time based on rotation speed
    assert sc.allclose(
        ch.relative_time_open(),
        sc.array(dims=['slit'], values=[180 * factor], unit='s'),
    )
    assert sc.allclose(
        ch.relative_time_close(),
        sc.array(dims=['slit'], values=[270 * factor], unit='s'),
    )
    assert sc.allclose(
        ch.open_duration(), sc.array(dims=['slit'], values=[90 * factor], unit='s')
    )


def test_relative_time_open_close_single_slit_clockwise_with_beam_position():
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(-7.21, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[90.0, 180.0], unit='deg'),
        beam_position=sc.scalar(-20.0, unit='deg'),
    )
    factor = (
        sc.constants.pi.value
        / 180  # to rad
        / (2 * sc.constants.pi.value)  # to angular frequency
        / abs(-7.21)
    )  # to time based on rotation speed
    assert sc.allclose(
        ch.relative_time_open(),
        sc.array(dims=['slit'], values=[110 * factor], unit='s'),
    )
    assert sc.allclose(
        ch.relative_time_close(),
        sc.array(dims=['slit'], values=[200 * factor], unit='s'),
    )
    assert sc.allclose(
        ch.open_duration(), sc.array(dims=['slit'], values=[90 * factor], unit='s')
    )


def test_relative_time_open_close_single_slit_anticlockwise_with_beam_position():
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(7.21, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[90.0, 180.0], unit='deg'),
        beam_position=sc.scalar(-20.0, unit='deg'),
    )
    factor = (
        sc.constants.pi.value
        / 180  # to rad
        / (2 * sc.constants.pi.value)  # to angular frequency
        / abs(7.21)
    )  # to time based on rotation speed
    assert sc.allclose(
        ch.relative_time_open(),
        sc.array(dims=['slit'], values=[160 * factor], unit='s'),
    )
    assert sc.allclose(
        ch.relative_time_close(),
        sc.array(dims=['slit'], values=[250 * factor], unit='s'),
    )
    assert sc.allclose(
        ch.open_duration(), sc.array(dims=['slit'], values=[90 * factor], unit='s')
    )


def test_relative_time_open_close_single_slit_across_tdc_clockwise():
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(-7.21, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[330.0, 380.0], unit='deg'),
    )
    factor = (
        sc.constants.pi.value
        / 180  # to rad
        / (2 * sc.constants.pi.value)  # to angular frequency
        / abs(-7.21)
    )  # to time based on rotation speed
    assert sc.allclose(
        ch.relative_time_open(),
        sc.array(dims=['slit'], values=[330 * factor], unit='s'),
    )
    assert sc.allclose(
        ch.relative_time_close(),
        sc.array(dims=['slit'], values=[380 * factor], unit='s'),
    )
    assert sc.allclose(
        ch.open_duration(), sc.array(dims=['slit'], values=[50 * factor], unit='s')
    )


def test_relative_time_open_close_single_slit_across_tdc_anticlockwise():
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(7.21, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[330.0, 380.0], unit='deg'),
    )
    factor = (
        sc.constants.pi.value
        / 180  # to rad
        / (2 * sc.constants.pi.value)  # to angular frequency
        / abs(7.21)
    )  # to time based on rotation speed
    assert sc.allclose(
        ch.relative_time_open(),
        sc.array(dims=['slit'], values=[-20 * factor], unit='s'),
    )
    assert sc.allclose(
        ch.relative_time_close(),
        sc.array(dims=['slit'], values=[30 * factor], unit='s'),
    )
    assert sc.allclose(
        ch.open_duration(), sc.array(dims=['slit'], values=[50 * factor], unit='s')
    )


def test_absolute_time_needs_tdc():
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(7.21, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[90.0, 180.0], unit='deg'),
    )
    with pytest.raises(RuntimeError):
        ch.time_open()
    with pytest.raises(RuntimeError):
        ch.time_close()


@pytest.mark.parametrize(
    'phase',
    (
        sc.scalar(0.0, unit='rad'),
        sc.scalar(1.2, unit='rad'),
        sc.scalar(-50.0, unit='deg'),
    ),
)
def test_absolute_time_open_close_single_slit_clockwise(phase):
    tdc = sc.datetimes(dims=['time'], values=[100, 200, 300], unit='ms')
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(-7.21, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[90.0, 180.0], unit='deg'),
        phase=phase,
        top_dead_center=tdc,
    )
    factor = (
        sc.constants.pi.value
        / 180  # to rad
        / (2 * sc.constants.pi.value)  # to angular frequency
        / abs(-7.21)
    )  # to time based on rotation speed
    assert sc.identical(
        ch.time_open(),
        sc.array(dims=['slit'], values=[1000 * 90 * factor], unit='ms', dtype=int)
        + tdc,
    )
    assert sc.identical(
        ch.time_close(),
        sc.array(dims=['slit'], values=[1000 * 180 * factor], unit='ms', dtype=int)
        + tdc,
    )


@pytest.mark.parametrize(
    'phase',
    (
        sc.scalar(0.0, unit='rad'),
        sc.scalar(1.2, unit='rad'),
        sc.scalar(-50.0, unit='deg'),
    ),
)
def test_absolute_time_open_close_single_slit_anticlockwise(phase):
    tdc = sc.datetimes(dims=['time'], values=[100, 200, 300], unit='ms')
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(7.21, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[90.0, 180.0], unit='deg'),
        phase=phase,
        top_dead_center=tdc,
    )
    factor = (
        sc.constants.pi.value
        / 180  # to rad
        / (2 * sc.constants.pi.value)  # to angular frequency
        / abs(7.21)
    )  # to time based on rotation speed
    assert sc.identical(
        ch.time_open(),
        sc.array(dims=['slit'], values=[1000 * 180 * factor], unit='ms', dtype=int)
        + tdc,
    )
    assert sc.identical(
        ch.time_close(),
        sc.array(dims=['slit'], values=[1000 * 270 * factor], unit='ms', dtype=int)
        + tdc,
    )


def test_absolute_time_open_close_single_slit_clockwise_with_delay():
    tdc = sc.datetimes(dims=['time'], values=[100, 200, 300], unit='ms')
    delay = sc.scalar(41002, unit='us')
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(-7.21, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[90.0, 180.0], unit='deg'),
        top_dead_center=tdc,
        delay=delay,
    )
    factor = (
        sc.constants.pi.value
        / 180  # to rad
        / (2 * sc.constants.pi.value)  # to angular frequency
        / abs(-7.21)
    )  # to time based on rotation speed
    assert sc.identical(
        ch.time_open(),
        sc.array(dims=['slit'], values=[1000 * 90 * factor], unit='ms', dtype=int)
        + tdc
        + delay.to(unit='ms'),
    )
    assert sc.identical(
        ch.time_close(),
        sc.array(dims=['slit'], values=[1000 * 180 * factor], unit='ms', dtype=int)
        + tdc
        + delay.to(unit='ms'),
    )


def test_absolute_time_open_close_single_slit_anticlockwise_with_delay():
    tdc = sc.datetimes(dims=['time'], values=[100, 200, 300], unit='ms')
    delay = sc.scalar(41002, unit='us')
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(7.21, unit='Hz'),
        slit_edges=sc.array(dims=['slit'], values=[90.0, 180.0], unit='deg'),
        top_dead_center=tdc,
        delay=delay,
    )
    factor = (
        sc.constants.pi.value
        / 180  # to rad
        / (2 * sc.constants.pi.value)  # to angular frequency
        / abs(7.21)
    )  # to time based on rotation speed
    assert sc.identical(
        ch.time_open(),
        sc.array(dims=['slit'], values=[1000 * 180 * factor], unit='ms', dtype=int)
        + tdc
        + delay.to(unit='ms'),
    )
    assert sc.identical(
        ch.time_close(),
        sc.array(dims=['slit'], values=[1000 * 270 * factor], unit='ms', dtype=int)
        + tdc
        + delay.to(unit='ms'),
    )


def test_absolute_time_open_close_two_slits_clockwise():
    tdc = sc.datetime(642, unit='s')
    delay = sc.scalar(4215, unit='ms')
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(9.31, unit='Hz'),
        slit_edges=sc.array(
            dims=['slit'], values=[30.0, 50.0, 170.0, 210.0], unit='deg'
        ),
        top_dead_center=tdc,
        delay=delay,
    )
    factor = (
        sc.constants.pi.value
        / 180  # to rad
        / (2 * sc.constants.pi.value)  # to angular frequency
        / abs(9.31)
    )  # to time based on rotation speed
    assert sc.identical(
        ch.time_open(),
        sc.datetimes(
            dims=['slit'],
            values=[int(30 * factor) + 642 + 4, int(170 * factor) + 642 + 4],
            unit='s',
        ),
    )
    assert sc.identical(
        ch.time_close(),
        sc.datetimes(
            dims=['slit'],
            values=[int(50 * factor) + 642 + 4, int(210 * factor) + 642 + 4],
            unit='s',
        ),
    )


def test_absolute_time_open_close_two_slits_anticlockwise():
    tdc = sc.datetime(642, unit='s')
    delay = sc.scalar(4215, unit='ms')
    ch = DiskChopper(
        position=sc.vector([0, 0, 0], unit='m'),
        rotation_speed=sc.scalar(-9.31, unit='Hz'),
        slit_edges=sc.array(
            dims=['slit'], values=[30.0, 50.0, 170.0, 210.0], unit='deg'
        ),
        top_dead_center=tdc,
        delay=delay,
    )
    factor = (
        sc.constants.pi.value
        / 180  # to rad
        / (2 * sc.constants.pi.value)  # to angular frequency
        / abs(-9.31)
    )  # to time based on rotation speed
    assert sc.identical(
        ch.time_open(),
        sc.datetimes(
            dims=['slit'],
            values=[int(310 * factor) + 642 + 4, int(150 * factor) + 642 + 4],
            unit='s',
        ),
    )
    assert sc.identical(
        ch.time_close(),
        sc.datetimes(
            dims=['slit'],
            values=[int(330 * factor) + 642 + 4, int(190 * factor) + 642 + 4],
            unit='s',
        ),
    )
