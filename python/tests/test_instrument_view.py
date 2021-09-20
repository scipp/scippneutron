import scippneutron as scn
import scipp as sc
import numpy as np
from .common import make_dataset_with_beamline


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
