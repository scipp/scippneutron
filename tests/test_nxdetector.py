from .test_nexus import open_nexus, open_json
from .nexus_helpers import NexusBuilder, Detector
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
        with pytest.raises(NotImplementedError):
            detector[...]


def test_loads_data_without_coords(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    builder = NexusBuilder()
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], values=[[1.1, 2.2], [3.3, 4.4]]))
    builder.add_detector(Detector(detector_numbers=np.array([1, 2, 3, 4]), data=da))
    with resource(builder)() as f:
        detector = nexus.NXroot(f, loader)['entry/detector_0']
        loaded = detector[...]
        assert sc.identical(loaded, da)


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
        assert sc.identical(loaded, da)
