import scippneutron as scn
import scipp as sc
import numpy as np
from .common import make_dataset_with_beamline
import pytest
import plopp as pp

import matplotlib

matplotlib.use('Agg')
pp.patch_scipp()


def test_neutron_instrument_view_3d():
    d = make_dataset_with_beamline()
    scn.instrument_view(d["a"])


def test_neutron_instrument_view_with_masks():
    d = make_dataset_with_beamline()
    x = np.transpose(d.coords['position'].values)[0, :]
    d['a'].masks['amask'] = sc.Variable(dims=['position'],
                                        values=np.less(np.abs(x), 0.5))
    scn.instrument_view(d["a"])


def test_neutron_instrument_view_with_cmap_args():
    d = make_dataset_with_beamline()
    scn.instrument_view(d["a"],
                        vmin=0.001 * sc.units.one,
                        vmax=5.0 * sc.units.one,
                        cmap="magma",
                        norm="log")


def _make_component_settings(*,
                             data,
                             center='sample_position',
                             type='box',
                             size_unit=sc.units.m,
                             wireframe=False,
                             component_size=(0.1, 0.1, 0.1)):
    comp_size = sc.vector(value=component_size, unit=size_unit) if isinstance(
        component_size, tuple) else sc.scalar(value=component_size, unit=size_unit)
    center = data.meta[center] if isinstance(center, str) else center
    sample_settings = {'center': center, 'size': comp_size, 'type': type}
    return sample_settings


def test_neutron_instrument_view_components_valid():
    d = make_dataset_with_beamline()
    scn.instrument_view(d["a"], components={'sample': _make_component_settings(data=d)})


def test_neutron_instrument_view_with_size_scalar():
    d = make_dataset_with_beamline()
    scn.instrument_view(
        d["a"],
        components={'sample': _make_component_settings(data=d, component_size=2.0)})


def test_neutron_instrument_view_components_multiple_valid():
    d = make_dataset_with_beamline()
    scn.instrument_view(d["a"],
                        components={
                            'sample':
                            _make_component_settings(data=d, center='sample_position'),
                            'source':
                            _make_component_settings(data=d, center='source_position')
                        })


def test_neutron_instrument_view_components_with_non_beamline_component():
    d = make_dataset_with_beamline()
    widget_center = sc.vector(value=[1, 1, 1], unit=sc.units.m)
    scn.instrument_view(
        d["a"],
        components={'widget': _make_component_settings(data=d, center=widget_center)})


def test_neutron_instrument_view_components_with_wireframe():
    d = make_dataset_with_beamline()
    scn.instrument_view(
        d["a"],
        components={'sample': _make_component_settings(data=d, wireframe=False)})
    scn.instrument_view(
        d["a"], components={'sample': _make_component_settings(data=d, wireframe=True)})


def test_neutron_instrument_view_components_with_invalid_type():
    d = make_dataset_with_beamline()
    # Check that all our valid shape types work
    for shape_type in ['box', 'cylinder', 'disk']:
        scn.instrument_view(
            d["a"],
            components={'sample': _make_component_settings(data=d, type=shape_type)})
    with pytest.raises(ValueError):
        scn.instrument_view(d["a"],
                            components={
                                'sample':
                                _make_component_settings(data=d, type='trefoil_knot')
                            })


def test_neutron_instrument_view_components_with_invalid_size_unit():
    d = make_dataset_with_beamline()
    # mm is fine, source_position is set in meters. Just a scaling factor.
    scn.instrument_view(
        d["a"],
        components={'sample': _make_component_settings(data=d, size_unit=sc.units.mm)})
    # cannot scale us to meters. This should throw
    with pytest.raises(sc.core.UnitError):
        scn.instrument_view(d["a"],
                            components={
                                'sample':
                                _make_component_settings(data=d, size_unit=sc.units.us)
                            })
