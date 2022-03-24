from .nexus_test import open_nexus, open_json
from .nexus_helpers import NexusBuilder
from typing import Callable
import scipp as sc
from scippneutron import nexus
from scippneutron.nexus import NX_class
import pytest


@pytest.fixture(params=[open_nexus, open_json])
def nexus_group(request):
    return request.param


def test_without_coords(nexus_group: Callable):
    signal = sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1, 2.2], [3.3, 4.4]])
    with nexus_group(NexusBuilder())() as f:
        entry = nexus.NXroot(f)['entry']
        data = entry.create_class('data1', NX_class.NXdata)
        data.create_field('signal', signal)
        data.attrs['axes'] = signal.dims
        data.attrs['signal'] = 'signal'
        assert sc.identical(data[...], sc.DataArray(signal))


def test_with_coords_matching_axis_names(nexus_group: Callable):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    with nexus_group(NexusBuilder())() as f:
        entry = nexus.NXroot(f)['entry']
        data = entry.create_class('data1', NX_class.NXdata)
        data.attrs['axes'] = da.dims
        data.attrs['signal'] = 'signal'
        data.create_field('signal', da.data)
        data.create_field('xx', da.coords['xx'])
        assert sc.identical(data[...], da)


def test_guessed_dim_for_coord_not_matching_axis_name(nexus_group: Callable):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx2'] = da.data['yy', 1]
    with nexus_group(NexusBuilder())() as f:
        entry = nexus.NXroot(f)['entry']
        data = entry.create_class('data1', NX_class.NXdata)
        data.attrs['axes'] = da.dims
        data.attrs['signal'] = 'signal'
        data.create_field('signal', da.data)
        data.create_field('xx2', da.coords['xx2'])
        assert sc.identical(data[...], da)


def test_multiple_coords(nexus_group: Callable):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    with nexus_group(NexusBuilder())() as f:
        entry = nexus.NXroot(f)['entry']
        data = entry.create_class('data1', NX_class.NXdata)
        data.attrs['axes'] = da.dims
        data.attrs['signal'] = 'signal'
        data.create_field('signal', da.data)
        data.create_field('xx', da.coords['xx'])
        data.create_field('xx2', da.coords['xx2'])
        data.create_field('yy', da.coords['yy'])
        assert sc.identical(data[...], da)


def test_slice_of_1d(nexus_group: Callable):
    da = sc.DataArray(sc.array(dims=['xx'], unit='m', values=[1, 2, 3]))
    da.coords['xx'] = da.data
    da.coords['xx2'] = da.data
    da.coords['scalar'] = sc.scalar(1.2)
    with nexus_group(NexusBuilder())() as f:
        entry = nexus.NXroot(f)['entry']
        data = entry.create_class('data1', NX_class.NXdata)
        data.attrs['axes'] = da.dims
        data.attrs['signal'] = 'signal'
        data.create_field('signal', da.data)
        data.create_field('xx', da.coords['xx'])
        data.create_field('xx2', da.coords['xx2'])
        data.create_field('scalar', da.coords['scalar'])
        assert sc.identical(data['xx', :2], da['xx', :2])
        assert sc.identical(data[:2], da['xx', :2])


def test_slice_of_multiple_coords(nexus_group: Callable):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    with nexus_group(NexusBuilder())() as f:
        entry = nexus.NXroot(f)['entry']
        data = entry.create_class('data1', NX_class.NXdata)
        data.attrs['axes'] = da.dims
        data.attrs['signal'] = 'signal'
        data.create_field('signal', da.data)
        data.create_field('xx', da.coords['xx'])
        data.create_field('xx2', da.coords['xx2'])
        data.create_field('yy', da.coords['yy'])
        assert sc.identical(data['xx', :2], da['xx', :2])


def test_guessed_dim_for_2d_coord_not_matching_axis_name(nexus_group: Callable):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx2'] = da.data
    with nexus_group(NexusBuilder())() as f:
        entry = nexus.NXroot(f)['entry']
        data = entry.create_class('data1', NX_class.NXdata)
        data.attrs['axes'] = da.dims
        data.attrs['signal'] = 'signal'
        data.create_field('signal', da.data)
        data.create_field('xx2', da.coords['xx2'])
        assert sc.identical(data[...], da)


def test_skips_axis_if_dim_guessing_finds_ambiguous_shape(nexus_group: Callable):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    da.coords['yy2'] = da.data['xx', 0]
    with nexus_group(NexusBuilder())() as f:
        entry = nexus.NXroot(f)['entry']
        data = entry.create_class('data1', NX_class.NXdata)
        data.attrs['axes'] = da.dims
        data.attrs['signal'] = 'signal'
        data.create_field('signal', da.data)
        data.create_field('yy2', da.coords['yy2'])
        da = data[...]
        assert 'yy2' not in da.coords


def test_guesses_transposed_dims_for_2d_coord(nexus_group: Callable):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx2'] = sc.transpose(da.data)
    with nexus_group(NexusBuilder())() as f:
        entry = nexus.NXroot(f)['entry']
        data = entry.create_class('data1', NX_class.NXdata)
        data.attrs['axes'] = da.dims
        data.attrs['signal'] = 'signal'
        data.create_field('signal', da.data)
        data.create_field('xx2', da.coords['xx2'])
        assert sc.identical(data[...], da)


@pytest.mark.parametrize("indices", [1, [1]], ids=['int', 'list-of-int'])
def test_indices_attribute_for_coord(nexus_group: Callable, indices):
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2], [4, 5]]))
    da.coords['yy2'] = da.data['xx', 0]
    with nexus_group(NexusBuilder())() as f:
        entry = nexus.NXroot(f)['entry']
        data = entry.create_class('data1', NX_class.NXdata)
        data.attrs['axes'] = da.dims
        data.attrs['signal'] = 'signal'
        data.attrs['yy2_indices'] = indices
        data.create_field('signal', da.data)
        data.create_field('yy2', da.coords['yy2'])
        assert sc.identical(data[...], da)


def test_transpose_indices_attribute_for_coord(nexus_group: Callable):
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2], [4, 5]]))
    da.coords['xx2'] = sc.transpose(da.data)
    with nexus_group(NexusBuilder())() as f:
        entry = nexus.NXroot(f)['entry']
        data = entry.create_class('data1', NX_class.NXdata)
        data.attrs['axes'] = da.dims
        data.attrs['signal'] = 'signal'
        data.attrs['xx2_indices'] = [1, 0]
        data.create_field('signal', da.data)
        data.create_field('xx2', da.coords['xx2'])
        assert sc.identical(data[...], da)


def test_auxiliary_signal_is_not_loaded_as_coord(nexus_group: Callable):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['xx', 0]
    with nexus_group(NexusBuilder())() as f:
        entry = nexus.NXroot(f)['entry']
        data = entry.create_class('data1', NX_class.NXdata)
        data.attrs['axes'] = da.dims
        data.attrs['signal'] = 'signal'
        # We flag 'xx' as auxiliary_signal. It should thus not be loaded as a coord,
        # even though we create the field.
        data.attrs['auxiliary_signals'] = ['xx']
        data.create_field('signal', da.data)
        data.create_field('xx', da.coords['xx'])
        del da.coords['xx']
        assert sc.identical(data[...], da)


def test_field_dims_match_NXdata_dims(nexus_group: Callable):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    with nexus_group(NexusBuilder())() as f:
        entry = nexus.NXroot(f)['entry']
        data = entry.create_class('data1', NX_class.NXdata)
        data.attrs['axes'] = da.dims
        data.attrs['signal'] = 'signal1'
        data.create_field('signal1', da.data)
        data.create_field('xx', da.coords['xx'])
        data.create_field('xx2', da.coords['xx2'])
        data.create_field('yy', da.coords['yy'])
        assert sc.identical(data['xx', :2].data, data['signal1']['xx', :2])
        assert sc.identical(data['xx', :2].coords['xx'], data['xx']['xx', :2])
        assert sc.identical(data['xx', :2].coords['xx2'], data['xx2']['xx', :2])
        assert sc.identical(data['xx', :2].coords['yy'], data['yy'][:])


def test_uses_default_field_dims_if_inference_fails(nexus_group: Callable):
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['yy2'] = sc.arange('yy', 4)
    with nexus_group(NexusBuilder())() as f:
        entry = nexus.NXroot(f)['entry']
        data = entry.create_class('data1', NX_class.NXdata)
        data.attrs['axes'] = da.dims
        data.attrs['signal'] = 'signal'
        data.create_field('signal', da.data)
        data.create_field('yy2', da.coords['yy2'])
        assert 'yy2' not in data[()].coords
        assert sc.identical(data['yy2'][()], da.coords['yy2'].rename(yy='dim_0'))


@pytest.mark.parametrize("unit", ['m', 's', None])
def test_create_field_from_variable(nexus_group: Callable, unit):
    var = sc.array(dims=['xx'], unit=unit, values=[3, 4])
    with nexus_group(NexusBuilder())() as f:
        entry = nexus.NXroot(f)['entry']
        entry.create_field('field', var)
        loaded = entry['field'][...]
        # Nexus does not support storing dim labels
        assert sc.identical(loaded, var.rename(xx=loaded.dim))


@pytest.mark.parametrize("nx_class", [NX_class.NXdata, NX_class.NXlog])
def test_create_class(nexus_group: Callable, nx_class):
    with nexus_group(NexusBuilder())() as f:
        entry = nexus.NXroot(f)['entry']
        group = entry.create_class('group', nx_class)
        assert group.nx_class == nx_class
