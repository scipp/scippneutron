from .nexus_test import open_nexus, open_json
from .nexus_helpers import NexusBuilder, Data
from typing import Callable, Tuple
import scipp as sc
from scippneutron.file_loading._nexus import LoadFromNexus
from scippneutron.file_loading._hdf5_nexus import LoadFromHdf5
from scippneutron.file_loading._json_nexus import LoadFromJson
from scippneutron import nexus
import pytest


@pytest.fixture(params=[(open_nexus, LoadFromHdf5()), (open_json, LoadFromJson(''))])
def nexus_group(request):
    return request.param


def test_without_coords(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1, 2.2], [3.3, 4.4]]))
    builder.add_data(Data(name='data1', data=da))
    with resource(builder)() as f:
        data = nexus.NXroot(f)['entry/data1']
        loaded = data[...]
        assert sc.identical(loaded, da)


def test_with_coords_matching_axis_names(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    builder.add_data(Data(name='data1', data=da))
    with resource(builder)() as f:
        data = nexus.NXroot(f)['entry/data1']
        loaded = data[...]
        assert sc.identical(loaded, da)


def test_guessed_dim_for_coord_not_matching_axis_name(
        nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx2'] = da.data['yy', 1]
    builder.add_data(Data(name='data1', data=da))
    with resource(builder)() as f:
        data = nexus.NXroot(f)['entry/data1']
        loaded = data[...]
        assert sc.identical(loaded, da)


def test_multiple_coords(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    builder.add_data(Data(name='data1', data=da))
    with resource(builder)() as f:
        data = nexus.NXroot(f)['entry/data1']
        loaded = data[...]
        assert sc.identical(loaded, da)


def test_slice_of_1d(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(sc.array(dims=['xx'], unit='m', values=[1, 2, 3]))
    da.coords['xx'] = da.data
    da.coords['xx2'] = da.data
    da.coords['scalar'] = sc.scalar(1.2)
    builder.add_data(Data(name='data1', data=da))
    with resource(builder)() as f:
        data = nexus.NXroot(f)['entry/data1']
        assert sc.identical(data['xx', :2], da['xx', :2])
        assert sc.identical(data[:2], da['xx', :2])


def test_slice_of_multiple_coords(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    builder.add_data(Data(name='data1', data=da))
    with resource(builder)() as f:
        data = nexus.NXroot(f)['entry/data1']
        assert sc.identical(data['xx', :2], da['xx', :2])


def test_guessed_dim_for_2d_coord_not_matching_axis_name(
        nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx2'] = da.data
    builder.add_data(Data(name='data1', data=da))
    with resource(builder)() as f:
        data = nexus.NXroot(f)['entry/data1']
        loaded = data[...]
        assert sc.identical(loaded, da)


def test_skips_axis_if_dim_guessing_finds_ambiguous_shape(
        nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    da.coords['yy2'] = da.data['xx', 0]
    builder.add_data(Data(name='data1', data=da))
    with resource(builder)() as f:
        data = nexus.NXroot(f)['entry/data1']
        da = data[...]
        assert 'yy2' not in da.coords


def test_guesses_transposed_dims_for_2d_coord(nexus_group: Tuple[Callable,
                                                                 LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx2'] = sc.transpose(da.data)
    builder.add_data(Data(name='data1', data=da))
    with resource(builder)() as f:
        data = nexus.NXroot(f)['entry/data1']
        loaded = data[...]
        assert sc.identical(loaded, da)


def test_integer_indices_attribute_for_coord(nexus_group: Tuple[Callable,
                                                                LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2], [4, 5]]))
    da.coords['yy2'] = da.data['xx', 0]
    builder.add_data(Data(name='data1', data=da, attrs={'yy2_indices': 1}))
    with resource(builder)() as f:
        data = nexus.NXroot(f)['entry/data1']
        loaded = data[...]
        assert sc.identical(loaded, da)


def test_list_indices_attribute_for_coord(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2], [4, 5]]))
    da.coords['yy2'] = da.data['xx', 0]
    builder.add_data(Data(name='data1', data=da, attrs={'yy2_indices': [1]}))
    with resource(builder)() as f:
        data = nexus.NXroot(f)['entry/data1']
        loaded = data[...]
        assert sc.identical(loaded, da)


def test_transpose_indices_attribute_for_coord(nexus_group: Tuple[Callable,
                                                                  LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2], [4, 5]]))
    da.coords['xx2'] = sc.transpose(da.data)
    builder.add_data(Data(name='data1', data=da, attrs={'xx2_indices': [1, 0]}))
    with resource(builder)() as f:
        data = nexus.NXroot(f)['entry/data1']
        loaded = data[...]
        assert sc.identical(loaded, da)


def test_auxiliary_signal_is_not_loaded_as_coord(nexus_group: Tuple[Callable,
                                                                    LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['xx', 0]
    # We flag 'xx' as auxiliary_signal. It should thus not be loaded as a coord.
    builder.add_data(Data(name='data1', data=da, attrs={'auxiliary_signals': ['xx']}))
    with resource(builder)() as f:
        data = nexus.NXroot(f)['entry/data1']
        loaded = data[...]
        del da.coords['xx']
        assert sc.identical(loaded, da)


def test_field_dims_match_NXdata_dims(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    builder.add_data(Data(name='data1', data=da))
    with resource(builder)() as f:
        data = nexus.NXroot(f)['entry/data1']
        assert sc.identical(data['xx', :2].data, data['signal1']['xx', :2])
        assert sc.identical(data['xx', :2].coords['xx'], data['xx']['xx', :2])
        assert sc.identical(data['xx', :2].coords['xx2'], data['xx2']['xx', :2])
        assert sc.identical(data['xx', :2].coords['yy'], data['yy'][:])


def test_uses_default_field_dims_if_inference_fails(nexus_group: Tuple[Callable,
                                                                       LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['yy2'] = sc.arange('yy', 4)
    builder.add_data(Data(name='data1', data=da))
    with resource(builder)() as f:
        data = nexus.NXroot(f)['entry/data1']
        assert 'yy2' not in data[()].coords
        assert sc.identical(data['yy2'][()], da.coords['yy2'].rename(yy='dim_0'))
