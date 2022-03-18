from .nexus_test import open_nexus, open_json
from .nexus_helpers import NexusBuilder, Detector, Transformation, TransformationType
import numpy as np
import pytest
from typing import Callable, Tuple
import scipp as sc
from scippneutron.file_loading._nexus import LoadFromNexus
from scippneutron.file_loading._hdf5_nexus import LoadFromHdf5
from scippneutron.file_loading._json_nexus import LoadFromJson
from scippneutron import nexus
from scippneutron.file_loading import nxtransformations


@pytest.fixture(params=[(open_nexus, LoadFromHdf5()), (open_json, LoadFromJson(''))])
def nexus_group(request):
    return request.param


def builder_with_detector(*, depends_on):
    builder = NexusBuilder()
    da = sc.DataArray(sc.array(dims=['xx', 'yy'], values=[[1.1, 2.2], [3.3, 4.4]]))
    detector_numbers = np.array([[1, 2], [3, 4]])
    builder.add_detector(
        Detector(detector_numbers=detector_numbers, data=da, depends_on=depends_on))
    return builder


def test_Transformation_with_single_value(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    offset = sc.spatial.translation(value=[1, 2, 3], unit='mm')
    vector = sc.vector(value=[0, 0, 1])
    value = sc.scalar(6.5, unit='mm')
    translation = Transformation(TransformationType.TRANSLATION,
                                 vector=vector.value,
                                 value=value.value,
                                 value_units=str(value.unit),
                                 offset=offset.values,
                                 offset_unit=str(offset.unit))
    builder = builder_with_detector(depends_on=translation)
    t = value.to(unit='m') * vector
    expected = sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit)
    expected = expected * sc.spatial.translation(value=[0.001, 0.002, 0.003], unit='m')
    with resource(builder)() as f:
        root = nexus.NXroot(f, loader)
        detector = root['entry/detector_0']
        depends_on = detector['depends_on'][()].value
        t = nxtransformations.Transformation(root[depends_on])
        assert t.depends_on is None
        assert sc.identical(t.offset, offset)
        assert sc.identical(t.vector, vector)
        assert sc.identical(t[()], expected)


def test_Transformation_with_multiple_values(nexus_group: Tuple[Callable,
                                                                LoadFromNexus]):
    resource, loader = nexus_group
    offset = sc.spatial.translation(value=[1, 2, 3], unit='m')
    vector = sc.vector(value=[0, 0, 1])
    log = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2], unit='m'),
        coords={'time': sc.array(dims=['time'], values=[11, 22], unit='s')})
    translation = Transformation(TransformationType.TRANSLATION,
                                 vector=vector.value,
                                 value=log.values,
                                 value_units=str(log.unit),
                                 time=log.coords['time'].values,
                                 time_units=str(log.coords['time'].unit),
                                 offset=offset.values,
                                 offset_unit=str(offset.unit))
    log.coords['time'] = sc.epoch(unit='ns') + log.coords['time'].to(unit='ns')
    builder = builder_with_detector(depends_on=translation)
    t = log * vector
    t.data = sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit)
    expected = t * offset
    with resource(builder)() as f:
        root = nexus.NXroot(f, loader)
        detector = root['entry/detector_0']
        depends_on = detector['depends_on'][()].value
        t = nxtransformations.Transformation(root[depends_on])
        assert t.depends_on is None
        assert sc.identical(t.offset, offset)
        assert sc.identical(t.vector, vector)
        assert sc.identical(t[()], expected)
