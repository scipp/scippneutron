# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Tuple

import numpy as np
import pytest
import scipp as sc

from scippneutron.conversion.graph import beamline
from scippneutron.tof import wfm
from scippneutron.tof.frames import _tof_from_wavelength

rng = np.random.default_rng()


def make_array(*,
               nevent: int = 1000,
               wav_min,
               wav_max,
               source_position: sc.Variable) -> sc.DataArray:
    wavelength = sc.array(dims=['event'],
                          values=rng.uniform(low=wav_min, high=wav_max, size=nevent),
                          unit='angstrom')
    events = sc.DataArray(sc.ones(sizes=wavelength.sizes),
                          coords={
                              'wavelength': wavelength,
                              'pixel': sc.arange('event', nevent) % 2
                          })
    da = events.group('pixel')
    da.coords['source_position'] = source_position
    da.coords['sample_position'] = sc.vector([1.0, 2.0, 3.0], unit='m')
    da.coords['position'] = sc.vectors(dims=['pixel'],
                                       values=[[1.0, 2.0, 5.0], [1.1, 2.2, 5.5]],
                                       unit='m')
    return da


def make_frame_data(*, source_position: sc.Variable, wav_min: sc.Variable,
                    wav_max: sc.Variable,
                    subframe_offset: sc.Variable) -> Tuple[sc.DataArray, sc.DataArray]:
    frame1 = make_array(nevent=1000,
                        wav_min=wav_min[0].values,
                        wav_max=wav_max[0].values,
                        source_position=source_position)
    frame2 = make_array(nevent=2000,
                        wav_min=wav_min[1].values,
                        wav_max=wav_max[1].values,
                        source_position=source_position)
    graph = beamline.beamline(scatter=True)
    graph['tof'] = _tof_from_wavelength
    frame1 = frame1.transform_coords('tof', graph=graph)
    frame2 = frame2.transform_coords('tof', graph=graph)
    tof_edges = sc.linspace('tof', 0.0, 100000.0, num=1000, unit='us')
    expected = frame1.hist(tof=tof_edges)
    expected += frame2.hist(tof=tof_edges)
    # Consistency check: Are all TOFs within edges?
    assert expected.sum().value == 3000.0
    frame1.bins.coords['tof'] += subframe_offset[0]
    frame2.bins.coords['tof'] += subframe_offset[1]
    # Consistency check: Params must be such that frames do not overlap
    assert sc.all(frame1.bins.coords['tof'].max() < frame2.bins.coords['tof'].min())
    frames = frame1.bins.concatenate(frame2)
    da = frames.drop_attrs(['source_position', 'incident_beam', 'L1', 'Ltotal'])
    del da.bins.attrs['wavelength']
    return da, expected


def test_x():
    source_position = sc.vector([1.0, 2.0, -70.0], unit='m')
    # note the overlap
    wav_min = sc.array(dims=['subframe'], values=[1.0, 2.9], unit='angstrom')
    wav_max = sc.array(dims=['subframe'], values=[3.0, 5.0], unit='angstrom')
    subframe_offset = sc.array(dims=['subframe'], values=[1234.0, 4567.0], unit='us')
    da, expected = make_frame_data(source_position=source_position,
                                   wav_min=wav_min,
                                   wav_max=wav_max,
                                   subframe_offset=subframe_offset)
    stitched = wfm.stitch(da,
                          wavelength_min=wav_min,
                          wavelength_max=wav_max,
                          subframe_source_position=1.0 * source_position,
                          subframe_offset=subframe_offset)
    hist = stitched.hist(tof=expected.coords['tof'])
    print(expected.data, hist.data)
    assert sc.identical(expected.data, hist.data)
    #da.coords['tof'] = sc.scalar(1.0, unit='ms')
    #with pytest.raises(ValueError):
    #    unwrap_frames(da,
    #                  scatter=True,
    #                  pulse_period=71.0 * sc.Unit('ms'),
    #                  frame_offset=30.1 * sc.Unit('ms'),
    #                  lambda_min=2.5 * sc.Unit('Angstrom'))
