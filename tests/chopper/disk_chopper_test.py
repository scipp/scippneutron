# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc

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
