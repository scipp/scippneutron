import scipp as sc
import numpy as np


def make_dataset_with_beamline():
    positions = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]]
    d = sc.Dataset(
        data={'a': sc.Variable(dims=['position', 'tof'], values=np.random.rand(4, 9))},
        coords={
            'tof':
            sc.Variable(dims=['tof'],
                        values=np.arange(1000.0, 1010.0),
                        unit=sc.units.us),
            'position':
            sc.vectors(dims=['position'], values=positions, unit=sc.units.m)
        })

    d.coords['source_position'] = sc.vector(value=np.array([0, 0, -10]),
                                            unit=sc.units.m)
    d.coords['sample_position'] = sc.vector(value=np.array([0, 0, 0]), unit=sc.units.m)
    return d
