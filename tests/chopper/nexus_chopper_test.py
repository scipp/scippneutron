# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc

from scippneutron.chopper import DiskChopperType, extract_chopper_from_nexus


@pytest.fixture()
def raw_nexus_chopper():
    return sc.DataGroup(
        {
            'type': 'Chopper type single',
            'position': sc.vector([0.0, 0.0, 2.0], unit='m'),
            'rotation_speed': sc.scalar(12.0, unit='Hz'),
            'beam_position': sc.scalar(45.0, unit='deg'),
            'phase': sc.scalar(-20.0, unit='deg'),
            'slit_edges': sc.array(
                dims=['dim_0'], values=[0.0, 60.0, 124.0, 126.0], unit='deg'
            ),
            'slit_height': sc.array(dims=['slit'], values=[0.4, 0.3], unit='m'),
            'radius': sc.scalar(0.5, unit='m'),
            'top_dead_center': sc.datetimes(
                dims=['time'], values=[12, 56, 78], unit='ms'
            ),
        }
    )


def test_post_process_assigns_default_type(raw_nexus_chopper):
    del raw_nexus_chopper['type']
    processed = extract_chopper_from_nexus(raw_nexus_chopper)
    assert processed['type'] == DiskChopperType.single


def test_post_process_extracts_from_logs(raw_nexus_chopper):
    raw_nexus_chopper['rotation_speed'] = sc.DataGroup(
        {
            'value': sc.array(dims=['time'], values=[12, 56, 78], unit='Hz'),
            'time': sc.datetimes(dims=['time'], values=[2, 5, 8], unit='s'),
        }
    )
    processed = extract_chopper_from_nexus(raw_nexus_chopper)
    assert sc.identical(
        processed['rotation_speed'],
        sc.array(dims=['time'], values=[12, 56, 78], unit='Hz'),
    )
