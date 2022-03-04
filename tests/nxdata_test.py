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


def test_guessed_dims_without_coords(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1.1, 2.2], [3.3, 4.4]]))
    builder.add_data(Data(name='data1', data=da))
    with resource(builder)() as f:
        data = nexus.NXroot(f, loader)['entry/data1']
        loaded = data[...]
        assert sc.identical(loaded, da)


def test_guessed_dims_with_coords(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    builder.add_data(Data(name='data1', data=da))
    with resource(builder)() as f:
        data = nexus.NXroot(f, loader)['entry/data1']
        loaded = data[...]
        assert sc.identical(loaded, da)


def test_guessed_dims_with_multiple_coords_per_dim(nexus_group: Tuple[Callable,
                                                                      LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(
        sc.array(dims=['xx', 'yy'], unit='m', values=[[1, 2, 3], [4, 5, 6]]))
    da.coords['xx'] = da.data['yy', 0]
    da.coords['xx2'] = da.data['yy', 1]
    da.coords['yy'] = da.data['xx', 0]
    builder.add_data(Data(name='data1', data=da))
    with resource(builder)() as f:
        data = nexus.NXroot(f, loader)['entry/data1']
        loaded = data[...]
        assert sc.identical(loaded, da)
