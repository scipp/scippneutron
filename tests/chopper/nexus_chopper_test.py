# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scippnexus as snx

from scippneutron.chopper import DiskChopper, DiskChopperType, NXDiskChopper

from ..externalfile import get_scippnexus_path


def test_from_nexus():
    path = get_scippnexus_path('2023/BIFROST_873855_00000015.hdf')
    with snx.File(path, 'r') as f:
        reference = f['entry']['instrument']['bandwidth_chopper_1'][()]
        reference['rotation_speed']['value'].unit = 'Hz'
    with snx.File(
        path,
        'r',
        definitions={**snx.base_definitions(), 'NXdisk_chopper': NXDiskChopper},
    ) as f:
        chopper_group = f['entry']['instrument']['bandwidth_chopper_1']
        ch = chopper_group[()]

    assert isinstance(ch, DiskChopper)
    assert ch.typ == DiskChopperType.single
    assert sc.identical(ch.rotation_speed, reference['rotation_speed']['value'])
    assert sc.identical(ch.delay, reference['delay']['value'])
    assert sc.identical(ch.radius, sc.scalar(0.35, unit='m'))
    assert ch.slits == 1
    assert sc.identical(ch.slit_height, sc.scalar(0.1, unit='m'))
    assert sc.identical(
        ch.slit_edges,
        sc.array(dims=['dim_0', 'edge'], values=[[0.0, 161.0]], unit='deg'),
    )
    # TODO position
