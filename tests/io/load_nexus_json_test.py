import json
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

from .nexus_helpers import NexusBuilder, Source, Stream


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
            'event_id': sc.array(dims=['event'], values=[1, 2, 3, 1, 3], unit=None),
            'event_time_offset': sc.array(
                dims=['event'], values=[456, 743, 347, 345, 632], unit='ns'
            ),
        },
    )
    event_time_zero = sc.datetimes(
        dims=['event_time_zero'],
        values=[
            1600766730000000000,
            1600766731000000000,
            1600766732000000000,
            1600766733000000000,
        ],
        unit='ns',
    )
    begin = sc.array(dims=['event'], values=[0, 3, 3, 5], unit=None)
    end = sc.array(dims=['event'], values=[3, 3, 5, 5], unit=None)
    binned = sc.bins(data=expected_events, dim='event', begin=begin, end=end)
    expected = sc.DataArray(
        binned.rename_dims({'event': 'event_time_zero'}),
        coords={'event_time_zero': event_time_zero},
    )

    sc.testing.assert_identical(loaded.coords['event_time_zero'], event_time_zero)
    # Using `loaded == expected` could fail if the order of events in the
    # bins is different.
    # Since the order is arbitrary, check that the bins have equal weights instead.
    assert sc.all(
        loaded.bins.concatenate(-expected).bins.sum().data
        == sc.scalar(0, unit='counts')
    )
