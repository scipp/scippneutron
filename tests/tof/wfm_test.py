# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import Tuple

import numpy as np
import scipp as sc

from scippneutron.conversion import graph
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
    g = graph.beamline.beamline(scatter=True)
    g['tof'] = _tof_from_wavelength
    frame1 = frame1.transform_coords('tof', graph=g)
    frame2 = frame2.transform_coords('tof', graph=g)
    tof_edges = sc.linspace('tof', 0.0, 100000.0, num=1001, unit='us')
    expected_wav = frame1.bins.concatenate(frame2)
    wav_edges = sc.linspace('wavelength',
                            wav_min.min().value,
                            wav_max.max().value,
                            num=1000,
                            unit='angstrom')
    expected_wav = expected_wav.hist(wavelength=wav_edges)
    expected = frame1.hist(tof=tof_edges)
    expected += frame2.hist(tof=tof_edges)
    # Consistency check: Are all TOFs within edges?
    assert expected.sum().value == 3000.0
    frame1.bins.coords['tof'] += subframe_offset[0]
    frame2.bins.coords['tof'] += subframe_offset[1]
    # Consistency check: Params must be such that frames do not overlap
    assert sc.all(frame1.bins.coords['tof'].max() < frame2.bins.coords['tof'].min())
    frames = frame1.bins.concatenate(frame2)
    da = frames.drop_attrs(
        ['source_position', 'incident_beam', 'scattered_beam', 'L1', 'L2', 'Ltotal'])
    del da.bins.attrs['wavelength']
    return da, expected_wav


def test_single_chopper_stitched_data_yields_correct_wavelength_histogram():
    source_position = sc.vector([1.0, 2.0, -70.0], unit='m')
    # note the overlap
    wav_min = sc.array(dims=['subframe'], values=[1.0, 2.9], unit='angstrom')
    wav_max = sc.array(dims=['subframe'], values=[3.0, 5.0], unit='angstrom')
    subframe_offset = sc.array(dims=['subframe'], values=[1234.0, 4567.0], unit='us')
    da, expected = make_frame_data(source_position=source_position,
                                   wav_min=wav_min,
                                   wav_max=wav_max,
                                   subframe_offset=subframe_offset)
    # No subframe overlap (for given wavelength), i.e., a single WFM chopper, same
    # subframe begin and end.
    stitched = wfm.stitch_elastic(da,
                                  wavelength_min=wav_min,
                                  wavelength_max=wav_max,
                                  subframe_begin_source_position=source_position,
                                  subframe_end_source_position=source_position,
                                  subframe_offset=subframe_offset)
    # The stitched result does not contain wavelength, but coords are setup to
    # yield correct result.
    stitched_wavelength = stitched.transform_coords('wavelength',
                                                    graph=graph.tof.kinematic("tof"))
    hist = stitched_wavelength.hist(wavelength=expected.coords['wavelength'])
    assert sc.identical(expected.data, hist.data)


def test_stitch_incorrect_given_bad_subframe_source_position():
    source_position = sc.vector([1.0, 2.0, -70.0], unit='m')
    # note the overlap
    wav_min = sc.array(dims=['subframe'], values=[1.0, 2.9], unit='angstrom')
    wav_max = sc.array(dims=['subframe'], values=[3.0, 5.0], unit='angstrom')
    subframe_offset = sc.array(dims=['subframe'], values=[1234.0, 4567.0], unit='us')
    da, expected = make_frame_data(source_position=source_position,
                                   wav_min=wav_min,
                                   wav_max=wav_max,
                                   subframe_offset=subframe_offset)
    stitched = wfm.stitch_elastic(da,
                                  wavelength_min=wav_min,
                                  wavelength_max=wav_max,
                                  subframe_begin_source_position=1.1 * source_position,
                                  subframe_end_source_position=1.1 * source_position,
                                  subframe_offset=subframe_offset)
    stitched_wavelength = stitched.transform_coords('wavelength',
                                                    graph=graph.tof.kinematic("tof"))
    hist = stitched_wavelength.hist(wavelength=expected.coords['wavelength'])
    assert not sc.identical(expected.data, hist.data)


def test_stitch_incorrect_given_bad_subframe_offset():
    source_position = sc.vector([1.0, 2.0, -70.0], unit='m')
    # note the overlap
    wav_min = sc.array(dims=['subframe'], values=[1.0, 2.9], unit='angstrom')
    wav_max = sc.array(dims=['subframe'], values=[3.0, 5.0], unit='angstrom')
    subframe_offset = sc.array(dims=['subframe'], values=[1234.0, 4567.0], unit='us')
    da, expected = make_frame_data(source_position=source_position,
                                   wav_min=wav_min,
                                   wav_max=wav_max,
                                   subframe_offset=subframe_offset)
    stitched = wfm.stitch_elastic(da,
                                  wavelength_min=wav_min,
                                  wavelength_max=wav_max,
                                  subframe_begin_source_position=source_position,
                                  subframe_end_source_position=source_position,
                                  subframe_offset=1.05 * subframe_offset)
    stitched_wavelength = stitched.transform_coords('wavelength',
                                                    graph=graph.tof.kinematic("tof"))
    hist = stitched_wavelength.hist(wavelength=expected.coords['wavelength'])
    assert not sc.identical(expected.data, hist.data)


def test_stitch_incorrect_given_bad_positions():
    source_position = sc.vector([1.0, 2.0, -70.0], unit='m')
    # note the overlap
    wav_min = sc.array(dims=['subframe'], values=[1.0, 2.9], unit='angstrom')
    wav_max = sc.array(dims=['subframe'], values=[3.0, 5.0], unit='angstrom')
    subframe_offset = sc.array(dims=['subframe'], values=[1234.0, 4567.0], unit='us')
    da, expected = make_frame_data(source_position=source_position,
                                   wav_min=wav_min,
                                   wav_max=wav_max,
                                   subframe_offset=subframe_offset)
    da.attrs['position'] = da.attrs['position']['pixel', 0]
    stitched = wfm.stitch_elastic(da,
                                  wavelength_min=wav_min,
                                  wavelength_max=wav_max,
                                  subframe_begin_source_position=source_position,
                                  subframe_end_source_position=source_position,
                                  subframe_offset=subframe_offset)
    stitched_wavelength = stitched.transform_coords('wavelength',
                                                    graph=graph.tof.kinematic("tof"))
    hist = stitched_wavelength.hist(wavelength=expected.coords['wavelength'])
    assert not sc.identical(expected.data, hist.data)


def test_stitch_incorrect_given_bad_wavelength_bounds():
    source_position = sc.vector([1.0, 2.0, -70.0], unit='m')
    # note the overlap
    wav_min = sc.array(dims=['subframe'], values=[1.0, 2.9], unit='angstrom')
    wav_max = sc.array(dims=['subframe'], values=[3.0, 5.0], unit='angstrom')
    subframe_offset = sc.array(dims=['subframe'], values=[1234.0, 4567.0], unit='us')
    da, expected = make_frame_data(source_position=source_position,
                                   wav_min=wav_min,
                                   wav_max=wav_max,
                                   subframe_offset=subframe_offset)
    wav_max[0] *= 0.9
    stitched = wfm.stitch_elastic(da,
                                  wavelength_min=wav_min,
                                  wavelength_max=wav_max,
                                  subframe_begin_source_position=source_position,
                                  subframe_end_source_position=source_position,
                                  subframe_offset=subframe_offset)
    stitched_wavelength = stitched.transform_coords('wavelength',
                                                    graph=graph.tof.kinematic("tof"))
    hist = stitched_wavelength.hist(wavelength=expected.coords['wavelength'])
    assert not sc.identical(expected.data, hist.data)
