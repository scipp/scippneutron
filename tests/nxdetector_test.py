from .nexus_test import open_nexus, open_json
from .nexus_helpers import NexusBuilder
import numpy as np
import pytest
import scipp as sc
import scippneutron as scn
from scippneutron import nexus
from scippneutron.nexus import NX_class


@pytest.fixture(params=[open_nexus, open_json])
def nexus_root(request):
    with request.param(NexusBuilder())() as f:
        yield nexus.NXroot(f)


@pytest.fixture(params=[open_nexus, open_json])
def nexus_group(request):
    return request.param


def test_raises_if_no_data_found(nexus_root):
    detector_numbers = sc.array(dims=[''], unit=None, values=np.array([1, 2, 3, 4]))
    detector = nexus_root.create_class('detector0', NX_class.NXdetector)
    detector.create_field('detector_numbers', detector_numbers)
    with pytest.raises(KeyError):
        detector[...]


def test_loads_events_when_data_and_events_found(nexus_root):
    detector_numbers = sc.array(dims=[''], unit=None, values=np.array([1, 2]))
    data = sc.ones(dims=['xx'], shape=[2])
    detector = nexus_root.create_class('detector0', NX_class.NXdetector)
    detector.create_field('detector_numbers', detector_numbers)
    detector.create_field('data', data)
    assert detector[...].bins is None
    detector.create_field('event_id', sc.array(dims=[''], unit=None, values=[1]))
    detector.create_field('event_time_offset', sc.array(dims=[''], unit='s',
                                                        values=[1]))
    detector.create_field('event_time_zero', sc.array(dims=[''], unit='s', values=[1]))
    detector.create_field('event_index', sc.array(dims=[''], unit='None', values=[0]))
    assert detector[...].bins is not None


def detector_numbers_xx_yy_1234():
    return sc.array(dims=['xx', 'yy'], unit=None, values=np.array([[1, 2], [3, 4]]))


def test_loads_data_without_coords(nexus_root):
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], values=[[1.1, 2.2], [3.3, 4.4]]))
    da.coords['detector_numbers'] = detector_numbers_xx_yy_1234()
    detector = nexus_root.create_class('detector0', NX_class.NXdetector)
    detector.create_field('detector_numbers', da.coords['detector_numbers'])
    detector.create_field('data', da.data)
    assert sc.identical(detector[...], da.rename_dims({'xx': 'dim_0', 'yy': 'dim_1'}))


def test_select_events_raises_if_detector_contains_data(nexus_root):
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], values=[[1.1, 2.2], [3.3, 4.4]]))
    da.coords['detector_numbers'] = detector_numbers_xx_yy_1234()
    detector = nexus_root.create_class('detector0', NX_class.NXdetector)
    detector.create_field('detector_numbers', da.coords['detector_numbers'])
    detector.create_field('data', da.data)
    with pytest.raises(nexus.NexusStructureError):
        detector.select_events


def test_loads_data_with_coords(nexus_root):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='K', values=[[1.1, 2.2], [3.3, 4.4]]))
    da.coords['detector_numbers'] = detector_numbers_xx_yy_1234()
    da.coords['xx'] = sc.array(dims=['xx'], unit='m', values=[0.1, 0.2])
    detector = nexus_root.create_class('detector0', NX_class.NXdetector)
    detector.create_field('detector_numbers', da.coords['detector_numbers'])
    detector.create_field('xx', da.coords['xx'])
    detector.create_field('data', da.data)
    detector.attrs['axes'] = ['xx', '.']
    assert sc.identical(detector[...], da.rename_dims({'yy': 'dim_1'}))


def create_event_data_ids_1234(group):
    group.create_field('event_id',
                       sc.array(dims=[''], unit=None, values=[1, 2, 4, 1, 2, 2]))
    group.create_field('event_time_offset',
                       sc.array(dims=[''], unit='s', values=[456, 7, 3, 345, 632, 23]))
    group.create_field('event_time_zero',
                       sc.array(dims=[''], unit='s', values=[1, 2, 3, 4]))
    group.create_field('event_index',
                       sc.array(dims=[''], unit='None', values=[0, 3, 3, 5]))


def test_loads_event_data_mapped_to_detector_numbers_based_on_their_event_id(
        nexus_root):
    detector_numbers = sc.array(dims=[''], unit=None, values=np.array([1, 2, 3, 4]))
    detector = nexus_root.create_class('detector0', NX_class.NXdetector)
    detector.create_field('detector_number', detector_numbers)
    create_event_data_ids_1234(detector.create_class('events', NX_class.NXevent_data))
    assert detector.dims == ['dim_0']
    assert detector.shape == (4, )
    loaded = detector[...]
    assert sc.identical(
        loaded.bins.size().data,
        sc.array(dims=['dim_0'], unit=None, dtype='int64', values=[2, 3, 0, 1]))
    assert 'event_time_offset' in loaded.bins.coords
    assert 'event_time_zero' in loaded.bins.coords


def test_loads_event_data_with_2d_detector_numbers(nexus_root):
    detector = nexus_root.create_class('detector0', NX_class.NXdetector)
    detector.create_field('detector_number', detector_numbers_xx_yy_1234())
    create_event_data_ids_1234(detector.create_class('events', NX_class.NXevent_data))
    assert detector.dims == ['dim_0', 'dim_1']
    assert detector.shape == (2, 2)
    loaded = detector[...]
    assert sc.identical(
        loaded.bins.size().data,
        sc.array(dims=['dim_0', 'dim_1'],
                 unit=None,
                 dtype='int64',
                 values=[[2, 3], [0, 1]]))


def test_select_events_slices_underlying_event_data(nexus_root):
    detector = nexus_root.create_class('detector0', NX_class.NXdetector)
    detector.create_field('detector_number', detector_numbers_xx_yy_1234())
    create_event_data_ids_1234(detector.create_class('events', NX_class.NXevent_data))
    assert sc.identical(
        detector.select_events['pulse', :2][...].bins.size().data,
        sc.array(dims=['dim_0', 'dim_1'],
                 unit=None,
                 dtype='int64',
                 values=[[1, 1], [0, 1]]))
    assert sc.identical(
        detector.select_events['pulse', :3][...].bins.size().data,
        sc.array(dims=['dim_0', 'dim_1'],
                 unit=None,
                 dtype='int64',
                 values=[[2, 2], [0, 1]]))
    assert sc.identical(
        detector.select_events['pulse', 3][...].bins.size().data,
        sc.array(dims=['dim_0', 'dim_1'],
                 unit=None,
                 dtype='int64',
                 values=[[0, 1], [0, 0]]))
    assert sc.identical(
        detector.select_events[...][...].bins.size().data,
        sc.array(dims=['dim_0', 'dim_1'],
                 unit=None,
                 dtype='int64',
                 values=[[2, 3], [0, 1]]))


def test_select_events_slice_does_not_affect_original_detector(nexus_root):
    detector = nexus_root.create_class('detector0', NX_class.NXdetector)
    detector.create_field('detector_number', detector_numbers_xx_yy_1234())
    create_event_data_ids_1234(detector.create_class('events', NX_class.NXevent_data))
    detector.select_events['pulse', 0][...]
    assert sc.identical(
        detector[...].bins.size().data,
        sc.array(dims=['dim_0', 'dim_1'],
                 unit=None,
                 dtype='int64',
                 values=[[2, 3], [0, 1]]))


def test_loading_event_data_creates_automatic_detector_numbers_if_not_present_in_file(
        nexus_root):
    detector = nexus_root.create_class('detector0', NX_class.NXdetector)
    create_event_data_ids_1234(detector.create_class('events', NX_class.NXevent_data))
    assert detector.dims == ['detector_number']
    with pytest.raises(nexus.NexusStructureError):
        detector.shape
    loaded = detector[...]
    assert sc.identical(
        loaded.bins.size().data,
        sc.array(dims=['detector_number'],
                 unit=None,
                 dtype='int64',
                 values=[2, 3, 0, 1]))


def test_loading_event_data_with_selection_and_automatic_detector_numbers_raises(
        nexus_root):
    detector = nexus_root.create_class('detector0', NX_class.NXdetector)
    create_event_data_ids_1234(detector.create_class('events', NX_class.NXevent_data))
    assert detector.dims == ['detector_number']
    with pytest.raises(nexus.NexusStructureError):
        detector['detector_number', 0]


def test_loading_event_data_with_full_selection_and_automatic_detector_numbers_works(
        nexus_root):
    detector = nexus_root.create_class('detector0', NX_class.NXdetector)
    create_event_data_ids_1234(detector.create_class('events', NX_class.NXevent_data))
    assert detector.dims == ['detector_number']
    assert detector[...].shape == [4]
    assert detector[()].shape == [4]


def test_can_load_nxdetector_from_bigfake():
    with nexus.File(scn.data.bigfake()) as f:
        da = f['entry/instrument/detector_1'][...]
        assert da.sizes == {'dim_0': 300, 'dim_1': 300}


def test_can_load_nxdetector_from_PG3():
    with nexus.File(scn.data.get_path('PG3_4844_event.nxs')) as f:
        det = f['entry/instrument/bank24']
        da = det[...]
        assert da.sizes == {'x_pixel_offset': 154, 'y_pixel_offset': 7}
        assert 'detector_number' not in da.coords
        assert da.coords['pixel_id'].sizes == da.sizes
        assert da.coords['distance'].sizes == da.sizes
        assert da.coords['polar_angle'].sizes == da.sizes
        assert da.coords['azimuthal_angle'].sizes == da.sizes
        assert da.coords['x_pixel_offset'].sizes == {'x_pixel_offset': 154}
        assert da.coords['y_pixel_offset'].sizes == {'y_pixel_offset': 7}
        # local_name is an example of a dataset with shape=[1] that is treated as scalar
        assert da.coords['local_name'].sizes == {}
        # Extra scalar fields not in underlying NXevent_data
        del da.coords['local_name']
        del da.coords['total_counts']
        assert sc.identical(da.sum(), det.events[()].sum())  # no event lost in binning


def test_event_data_field_dims_labels(nexus_root):
    detector_numbers = sc.array(dims=[''], unit=None, values=np.array([1, 2, 3, 4]))
    detector = nexus_root.create_class('detector0', NX_class.NXdetector)
    detector.create_field('detector_number', detector_numbers)
    create_event_data_ids_1234(detector.create_class('events', NX_class.NXevent_data))
    assert detector['detector_number'].dims == ['dim_0']


def test_nxevent_data_selection_yields_correct_pulses(nexus_root):
    detector = nexus_root.create_class('detector0', NX_class.NXdetector)
    create_event_data_ids_1234(detector.create_class('events', NX_class.NXevent_data))

    class Load:
        def __getitem__(self, select=...):
            da = detector['events'][select]
            return da.bins.size().values

    assert np.array_equal(Load()[...], [3, 0, 2, 1])
    assert np.array_equal(Load()['pulse', 0], 3)
    assert np.array_equal(Load()['pulse', 1], 0)
    assert np.array_equal(Load()['pulse', 3], 1)
    assert np.array_equal(Load()['pulse', -1], 1)
    assert np.array_equal(Load()['pulse', -2], 2)
    assert np.array_equal(Load()['pulse', 0:0], [])
    assert np.array_equal(Load()['pulse', 1:1], [])
    assert np.array_equal(Load()['pulse', 1:-3], [])
    assert np.array_equal(Load()['pulse', 3:3], [])
    assert np.array_equal(Load()['pulse', -1:-1], [])
    assert np.array_equal(Load()['pulse', 0:1], [3])
    assert np.array_equal(Load()['pulse', 0:-3], [3])
    assert np.array_equal(Load()['pulse', -1:], [1])
    assert np.array_equal(Load()['pulse', -2:-1], [2])
    assert np.array_equal(Load()['pulse', -2:], [2, 1])
    assert np.array_equal(Load()['pulse', :-2], [3, 0])
