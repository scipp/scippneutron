from .test_nexus import open_nexus, open_json
from .nexus_helpers import NexusBuilder, Detector, EventData
import numpy as np
import pytest
from typing import Callable, Tuple
import scipp as sc
from scippneutron.file_loading._nexus import LoadFromNexus
from scippneutron.file_loading._hdf5_nexus import LoadFromHdf5
from scippneutron.file_loading._json_nexus import LoadFromJson
from scippneutron import nexus


@pytest.fixture(params=[(open_nexus, LoadFromHdf5()), (open_json, LoadFromJson(''))])
def nexus_group(request):
    return request.param


def test_raises_if_no_data_found(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    builder.add_detector(Detector(detector_numbers=np.array([1, 2, 3, 4])))
    with resource(builder)() as f:
        detector = nexus.NXroot(f, loader)['entry/detector_0']
        with pytest.raises(KeyError):
            detector[...]


def test_raises_if_data_and_event_data_found(nexus_group: Tuple[Callable,
                                                                LoadFromNexus]):
    resource, loader = nexus_group
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], values=[[1.1, 2.2], [3.3, 4.4]]))
    event_data = EventData(
        event_id=np.array([1, 2, 4, 1, 2, 2]),
        event_time_offset=np.array([456, 743, 347, 345, 632, 23]),
        event_time_zero=np.array([1, 2, 3, 4]),
        event_index=np.array([0, 3, 3, 5]),
    )
    builder = NexusBuilder()
    builder.add_detector(
        Detector(detector_numbers=np.array([1, 2, 3, 4]),
                 data=da,
                 event_data=event_data))
    with resource(builder)() as f:
        detector = nexus.NXroot(f, loader)['entry/detector_0']
        with pytest.raises(nexus.NexusStructureError):
            detector[...]


def test_loads_data_without_coords(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], values=[[1.1, 2.2], [3.3, 4.4]]))
    builder.add_detector(Detector(detector_numbers=np.array([1, 2, 3, 4]), data=da))
    with resource(builder)() as f:
        detector = nexus.NXroot(f, loader)['entry/detector_0']
        loaded = detector[...]
        assert sc.identical(loaded, da.rename_dims({'xx': 'dim_0', 'yy': 'dim_1'}))


def test_loads_data_with_coords(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='K', values=[[1.1, 2.2], [3.3, 4.4]]))
    da.coords['xx'] = sc.array(dims=['xx'], unit='m', values=[0.1, 0.2])
    builder.add_detector(Detector(detector_numbers=np.array([1, 2, 3, 4]), data=da))
    with resource(builder)() as f:
        detector = nexus.NXroot(f, loader)['entry/detector_0']
        loaded = detector[...]
        assert sc.identical(loaded, da.rename_dims({'yy': 'dim_1'}))


def test_loads_event_data_mapped_to_detector_numbers_based_on_their_event_id(
        nexus_group: Tuple[Callable, LoadFromNexus]):
    event_time_offsets = np.array([456, 743, 347, 345, 632, 23])
    event_data = EventData(
        event_id=np.array([1, 2, 3, 1, 2, 2]),
        event_time_offset=event_time_offsets,
        event_time_zero=np.array([1, 2, 3, 4]),
        event_index=np.array([0, 3, 3, 5]),
    )
    builder = NexusBuilder()
    builder.add_detector(
        Detector(detector_numbers=np.array([1, 2, 3, 4]), event_data=event_data))
    resource, loader = nexus_group
    with resource(builder)() as f:
        detector = nexus.NXroot(f, loader)['entry/detector_0']
        assert detector.dims == ['detector_number']
        assert detector.shape == (4, )
        loaded = detector[...]
        assert sc.identical(
            loaded.bins.size().data,
            sc.array(dims=['detector_number'], dtype='int64', values=[2, 3, 1, 0]))


def test_loading_event_data_creates_automatic_detector_numbers_if_not_present_in_file(
        nexus_group: Tuple[Callable, LoadFromNexus]):
    event_time_offsets = np.array([456, 743, 347, 345, 632, 23])
    event_data = EventData(
        event_id=np.array([1, 2, 4, 1, 2, 2]),
        event_time_offset=event_time_offsets,
        event_time_zero=np.array([1, 2, 3]),
        event_index=np.array([0, 3, 5]),
    )
    builder = NexusBuilder()
    builder.add_detector(Detector(event_data=event_data))
    resource, loader = nexus_group
    with resource(builder)() as f:
        detector = nexus.NXroot(f, loader)['entry/detector_0']
        assert detector.dims == ['detector_number']
        with pytest.raises(nexus.NexusStructureError):
            assert detector.shape == (4, )
        loaded = detector[...]
        assert sc.identical(
            loaded.bins.size().data,
            sc.array(dims=['detector_number'], dtype='int64', values=[2, 3, 0, 1]))
