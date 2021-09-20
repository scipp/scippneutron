import scippneutron as scn
import scipp as sc
import numpy as np
from .common import make_dataset_with_beamline
import pytest


def test_neutron_instrument_view_3d():
    d = make_dataset_with_beamline()
    scn.instrument_view(d["a"])


def test_neutron_instrument_view_with_dataset():
    d = make_dataset_with_beamline()
    d['b'] = sc.Variable(dims=['position', 'tof'], values=np.arange(36.).reshape(4, 9))
    scn.instrument_view(d)


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
                             center='sample_position',
                             type='box',
                             size_unit=sc.units.m,
                             wireframe=False,
                             component_size=(0.1, 0.1, 0.1)):
    comp_size = sc.vector(value=component_size, unit=size_unit) if isinstance(
        component_size, tuple) else sc.scalar(value=component_size, unit=size_unit)
    sample_settings = {'center': center, 'size': comp_size, 'type': type}
    return sample_settings


def test_neutron_instrument_view_components_valid():
    d = make_dataset_with_beamline()
    scn.instrument_view(d["a"], components={'sample': _make_component_settings()})


def test_neutron_instrument_view_with_size_scalar():
    d = make_dataset_with_beamline()
    scn.instrument_view(
        d["a"], components={'sample': _make_component_settings(component_size=2.0)})


def test_neutron_instrument_view_components_multiple_valid():
    d = make_dataset_with_beamline()
    scn.instrument_view(d["a"],
                        components={
                            'sample':
                            _make_component_settings(center='sample_position'),
                            'source': _make_component_settings(center='source_position')
                        })


def test_neutron_instrument_view_components_with_wireframe():
    d = make_dataset_with_beamline()
    scn.instrument_view(
        d["a"], components={'sample': _make_component_settings(wireframe=False)})
    scn.instrument_view(d["a"],
                        components={'sample': _make_component_settings(wireframe=True)})


def test_neutron_instrument_view_components_with_invalid_type():
    d = make_dataset_with_beamline()
    with pytest.raises(ValueError):
        scn.instrument_view(
            d["a"],
            components={'sample': _make_component_settings(type='trefoil_knot')})


def test_neutron_instrument_view_components_with_invalid_size_unit():
    d = make_dataset_with_beamline()
    # mm is fine, source_position is set in meters. Just a scaling factor.
    scn.instrument_view(
        d["a"], components={'sample': _make_component_settings(size_unit=sc.units.mm)})
    # cannot scale us to meters. This should throw
    with pytest.raises(sc.core.UnitError):
        scn.instrument_view(
            d["a"],
            components={'sample': _make_component_settings(size_unit=sc.units.us)})
