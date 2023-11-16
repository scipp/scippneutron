# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import io
import pytest
import scipp as sc
import scippnexus as snx
import numpy as np

from scippneutron.chopper import DiskChopper, DiskChopperType, NXDiskChopper

@pytest.fixture
def chopper_nexus_file() -> io.BytesIO:
    rng = np.random.default_rng(81471)

    buffer = io.BytesIO()
    with snx.File(buffer, 'w') as f:
        entry = f.create_class('entry', 'NXentry')
        instrument = entry.create_class('instrument', 'NXinstrument')
        chopper = instrument.create_class('chopper', 'NXdisk_chopper')

        chopper.create_field('radius', sc.scalar(0.35, unit='m'))
        chopper.create_field('slits', sc.index(2))
        chopper.create_field('slit_height', sc.array(dims=['dim'], values=[0.1, 0.12], unit='m'))
        chopper.create_field('slit_edges', sc.array(dims=['dim'], values=[10.0, 160.0, 210.0, 280.0], unit='deg'))

        rotation_speed = chopper.create_class('rotation_speed', 'NXlog')
        time = sc.arange('t', sc.datetime('2020-06-09T13:14:09'), sc.datetime('2020-06-09T13:16:24'), unit='s').to(unit='us')
        rotation_speed.create_field('value', sc.array(dims=['t'], values=rng.uniform(13.5, 14.5, len(time)), unit='Hz'))
        rotation_speed.create_field('time', time)

        delay = chopper.create_class('delay', 'NXlog')
        delay.create_field('value', sc.array(dims=['t'], values=[0.04], unit='s'))
        delay.create_field('time', sc.datetimes(dims=['t'], values=['2020-06-09T13:16:09'], unit='us'))

    return buffer


def test_from_nexus(chopper_nexus_file):
    with snx.File(chopper_nexus_file, 'r') as f:
        reference = f['entry']['instrument']['chopper'][()]
    with snx.File(
        chopper_nexus_file,
        'r',
        definitions={**snx.base_definitions(), 'NXdisk_chopper': NXDiskChopper},
    ) as f:
        chopper_group = f['entry']['instrument']['chopper']
        ch = chopper_group[()]

    assert isinstance(ch, DiskChopper)
    assert ch.typ == DiskChopperType.single
    assert sc.identical(ch.rotation_speed, reference['rotation_speed'])
    assert sc.identical(ch.delay, reference['delay'])
    assert sc.identical(ch.radius, sc.scalar(0.35, unit='m'))
    assert ch.slits == 2
    assert sc.identical(ch.slit_height, sc.array(dims=['dim_0'], values=[0.1, 0.12], unit='m'))
    assert sc.identical(
        ch.slit_edges,
        sc.array(dims=['dim_0', 'edge'], values=[[10.0, 160.0], [210.0, 280.0]], unit='deg'),
    )
    # TODO position
