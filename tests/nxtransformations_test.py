from .nexus_test import open_nexus, open_json
from .nexus_helpers import NexusBuilder
import numpy as np
import pytest
import scipp as sc
from scippneutron import nexus
from scippneutron.nexus import NX_class
from scippneutron.file_loading import nxtransformations


@pytest.fixture(params=[open_nexus, open_json])
def nxroot(request):
    with request.param(NexusBuilder())() as f:
        yield nexus.NXroot(f)


def create_detector(group):
    data = sc.array(dims=['xx', 'yy'], values=[[1.1, 2.2], [3.3, 4.4]])
    detector_numbers = sc.array(dims=['xx', 'yy'],
                                unit=None,
                                values=np.array([[1, 2], [3, 4]]))
    detector = group.create_class('detector_0', NX_class.NXdetector)
    detector.create_field('detector_number', detector_numbers)
    detector.create_field('data', data)
    return detector


def test_Transformation_with_single_value(nxroot):
    detector = create_detector(nxroot)
    detector.create_field('depends_on', sc.scalar('/detector_0/transformations/t1'))
    transformations = detector.create_class('transformations',
                                            NX_class.NXtransformations)
    value = sc.scalar(6.5, unit='mm')
    offset = sc.spatial.translation(value=[1, 2, 3], unit='mm')
    vector = sc.vector(value=[0, 0, 1])
    t = value.to(unit='m') * vector
    expected = sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit)
    expected = expected * sc.spatial.translation(value=[0.001, 0.002, 0.003], unit='m')
    value = transformations.create_field('t1', value)
    value.attrs['depends_on'] = '.'
    value.attrs['transformation_type'] = 'translation'
    value.attrs['offset'] = offset.values
    value.attrs['offset_units'] = str(offset.unit)
    value.attrs['vector'] = vector.value

    depends_on = detector['depends_on'][()].value
    t = nxtransformations.Transformation(nxroot[depends_on])
    assert t.depends_on is None
    assert sc.identical(t.offset, offset)
    assert sc.identical(t.vector, vector)
    assert sc.identical(t[()], expected)


def test_Transformation_with_multiple_values(nxroot):
    detector = create_detector(nxroot)
    detector.create_field('depends_on', sc.scalar('/detector_0/transformations/t1'))
    transformations = detector.create_class('transformations',
                                            NX_class.NXtransformations)
    log = sc.DataArray(
        sc.array(dims=['time'], values=[1.1, 2.2], unit='m'),
        coords={'time': sc.array(dims=['time'], values=[11, 22], unit='s')})
    log.coords['time'] = sc.epoch(unit='ns') + log.coords['time'].to(unit='ns')
    offset = sc.spatial.translation(value=[1, 2, 3], unit='m')
    vector = sc.vector(value=[0, 0, 1])
    t = log * vector
    t.data = sc.spatial.translations(dims=t.dims, values=t.values, unit=t.unit)
    expected = t * offset
    value = transformations.create_class('t1', NX_class.NXlog)
    value['time'] = log.coords['time'] - sc.epoch(unit='ns')
    value['value'] = log.data
    value.attrs['depends_on'] = '.'
    value.attrs['transformation_type'] = 'translation'
    value.attrs['offset'] = offset.values
    value.attrs['offset_units'] = str(offset.unit)
    value.attrs['vector'] = vector.value

    depends_on = detector['depends_on'][()].value
    t = nxtransformations.Transformation(nxroot[depends_on])
    assert t.depends_on is None
    assert sc.identical(t.offset, offset)
    assert sc.identical(t.vector, vector)
    assert sc.identical(t[()], expected)
