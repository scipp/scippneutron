import json
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipp as sc
import scipp.testing
import scippnexus as snx

from scippneutron.io.nexus.load_nexus import json_nexus_group, load_nexus_json_str

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nexus_helpers import NexusBuilder, Source, Stream


class JsonNexusExampleLoader:
    BASE_PATH = Path(__file__).resolve().parent / "json_nexus_examples"

    @staticmethod
    @lru_cache
    def _load(path: Path) -> dict[str, Any]:
        with path.open('r') as f:
            return json.load(f)

    def load(self, name: str) -> dict[str, Any]:
        return deepcopy(self._load(self.BASE_PATH.joinpath(name).with_suffix('.json')))

    def __getattr__(self, name: str) -> dict[str, Any]:
        try:
            return self.load(name)
        except FileNotFoundError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from None


@pytest.fixture(scope="module")
def examples():
    return JsonNexusExampleLoader()


def make_group(json_dict: dict[str, Any]) -> snx.Group:
    return json_nexus_group({'children': [json_dict]})


@pytest.mark.skip("TODO Stream handling with log not implemented")
def test_stream_object_as_transformation_results_in_warning():
    builder = NexusBuilder()
    builder.add_component(Source("source"))
    stream_path = "/entry/streamed_nxlog_transform"
    builder.add_stream(Stream(stream_path))
    builder.add_dataset_at_path("/entry/source/depends_on", stream_path, {})

    with pytest.warns(UserWarning):
        loaded_data, _ = load_nexus_json_str(builder.json_string)

    # A 0 distance translation is used in place of the streamed transformation
    default = [0, 0, 0]
    assert np.allclose(loaded_data["source_position"].values, default)
    assert loaded_data["source_position"].unit == sc.Unit("m")


def test_nexus_json_load_dataset_in_entry(examples):
    group = make_group(examples.entry)
    assert group[()]['entry']['title'] == 'my experiment'
    assert group['entry'][()]['title'] == 'my experiment'
    assert group['entry']['title'][()] == 'my experiment'


def test_nexus_json_load_event_data(examples):
    entry = examples.entry
    entry['children'].append(examples.event_data)
    group = make_group(entry)
    loaded = group[()]['entry']['events_0']

    expected_events = sc.DataArray(
        sc.ones(sizes={'event': 5}, unit='counts', dtype='float32'),
        coords={
            'event_id': sc.array(
                dims=['event'], values=[1, 2, 3, 1, 3], unit=None, dtype='int64'
            ),
            'event_time_offset': sc.array(
                dims=['event'],
                values=[456, 743, 347, 345, 632],
                unit='ns',
                dtype='int64',
            ),
        },
    )
    event_time_zero = sc.datetimes(
        dims=['event_time_zero'],
        values=[
            1600773930000000000,
            1600773931000000000,
            1600773932000000000,
            1600773933000000000,
        ],
        unit='ns',
    )
    begin = sc.array(dims=['event'], values=[0, 3, 3, 5], unit=None, dtype='int64')
    end = sc.array(dims=['event'], values=[3, 3, 5, 5], unit=None, dtype='int64')
    binned = sc.bins(data=expected_events, dim='event', begin=begin, end=end)
    expected = sc.DataArray(
        binned.rename_dims({'event': 'event_time_zero'}),
        coords={'event_time_zero': event_time_zero},
    )

    sc.testing.assert_identical(loaded, expected)


def test_nexus_json_load_detector_with_event_data(examples):
    detector = examples.detector
    detector['children'].append(examples.event_data)
    entry = examples.entry
    entry['children'].append(detector)
    group = make_group(entry)
    loaded = group[()]['entry']['detector_0']

    expected_events = sc.DataArray(
        sc.ones(sizes={'event': 5}, unit='counts', dtype='float32'),
        coords={
            'event_id': sc.array(
                dims=['event'], values=[1, 2, 3, 1, 3], unit=None, dtype='int64'
            ),
            'event_time_zero': sc.datetimes(
                dims=['event'],
                values=[
                    1600773930000000000,
                    1600773930000000000,
                    1600773930000000000,
                    1600773932000000000,
                    1600773932000000000,
                ],
                unit='ns',
            ),
            'event_time_offset': sc.array(
                dims=['event'],
                values=[456, 743, 347, 345, 632],
                unit='ns',
                dtype='int64',
            ),
        },
    )
    detector_number = sc.array(
        dims=['detector_number'], values=[1, 2, 3, 4], unit=None, dtype='int64'
    )
    expected = expected_events.group(
        sc.array(dims=['event_id'], values=[1, 2, 3, 4], unit=None, dtype='int64')
    )
    expected = expected.rename({'event_id': 'detector_number'}).assign_coords(
        detector_number=detector_number
    )

    assert loaded.keys() == {'events_0'}
    sc.testing.assert_identical(loaded['events_0'], expected)


def test_nexus_json_load_log(examples):
    entry = examples.entry
    entry['children'].append(examples.log)
    group = make_group(entry)
    loaded = group[()]['entry']['test_log']

    expected = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2, 3.3], unit='m'),
        coords={
            'time': sc.datetimes(
                dims=['time'],
                values=[1354291220000000000, 1354291231000000000, 1354291242000000000],
                unit='ns',
            )
        },
    )
    sc.testing.assert_identical(loaded, expected)


def test_nexus_json_load_log_utf8_unit(examples):
    log = examples.log
    assert log['children'][0]['config']['name'] == 'value', 'is the expected child'
    assert log['children'][0]['attributes'][0]['name'] == 'units'
    log['children'][0]['attributes'][0]['values'] = '\u00b0'  # '°', i.e., degrees

    entry = examples.entry
    entry['children'].append(log)
    group = make_group(entry)
    loaded = group[()]['entry']['test_log']

    expected = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2, 3.3], unit='deg'),
        coords={
            'time': sc.datetimes(
                dims=['time'],
                values=[1354291220000000000, 1354291231000000000, 1354291242000000000],
                unit='ns',
            )
        },
    )
    sc.testing.assert_identical(loaded, expected)


# TODO: remove warning filter
# Loading a NXmonitor fails with
# 'Signal is not array like' error and issues a warning.
# But this should probably be fixed elsewhere.
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_nexus_json_load_ymir_instrument(examples):
    make_group(examples.instrument)[()]


def test_nexus_json_load_dataset(examples):
    dg = make_group(examples.dataset)[()]
    assert dg['name'] == 'YMIR'


def test_nexus_json_load_array_dataset(examples):
    dg = make_group(examples.array_dataset)[()]
    sc.testing.assert_identical(
        dg['slit_edges'],
        sc.array(dims=['dim_0'], values=[0.0, 15.0, 180.0, 195.0], unit='deg'),
    )
