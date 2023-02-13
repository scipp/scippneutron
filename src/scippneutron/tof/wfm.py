# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
Frame transformations for time-of-flight neutron scattering data.
"""
import uuid

import scipp as sc
from scipp.constants import pi

from ..conversion.graph.beamline import Ltotal
from .frames import _tof_from_wavelength


def subframe_time_bounds_from_wavelengths(wavelength_min: sc.Variable,
                                          wavelength_max: sc.Variable,
                                          sample_position: sc.Variable, L2: sc.Variable,
                                          subframe_offset,
                                          subframe_source_position) -> sc.Variable:
    dim = subframe_offset.dim
    dummy = uuid.uuid4().hex
    L1 = sc.norm(sample_position - subframe_source_position)
    Ltotal = L1 + L2
    wavelength = sc.concat([wavelength_min, wavelength_max], dummy)
    time_bounds = subframe_offset + _tof_from_wavelength(wavelength=wavelength,
                                                         Ltotal=Ltotal)
    return time_bounds.transpose([dim, dummy]).flatten(to='tof')


def stitch(da: sc.DataArray, subframe_start: sc.Variable, subframe_stop: sc.Variable,
           subframe_offset: sc.Variable) -> sc.DataArray:
    dim = 'tof'
    # subframe_start and subframe_stop are pixel-dependent in general
    edges = sc.concat([subframe_start, subframe_stop], 'frame')
    edges['frame', ::2] = subframe_start
    edges['frame', 1::2] = subframe_stop
    edges = edges.rename(frame=dim)
    print(edges)
    binned = da.bin({dim: edges})
    binned.bins.coords[dim][dim, ::2] -= subframe_offset.rename(frame=dim)
    del binned.coords[dim]
    return binned[dim, ::2].bins.concat(dim)


def xxx(lambda_min: sc.Variable, lambda_max: sc.Variable):
    # wrt source pulse stop
    t_min = _tof_from_wavelength(wavelength=lambda_min, Ltotal=Ltotal)
    # wrt source pulse start
    t_max = _tof_from_wavelength(wavelength=lambda_max, Ltotal=Ltotal)
    # tof in unwrapped pulses is wrt source pulse stop (given by frame_offset)
    # or more precisely, it should be the offset of the emission time of the fastest
    # neutrons that can pass through the choppers
    # better convert to time_since_pulse = tof + frame_offset, since chopper phases
    # are defined based on that.
    t_max -= pulse_duration


def setup_chopper(chopper: sc.DataGroup, phase: sc.Variable,
                  rotation_speed: sc.Variable) -> sc.DataGroup:
    chopper = chopper.copy()
    chopper['phase'] = phase
    # TODO I don't know if this is the correct way of handling this
    chopper['rotation_speed'] = rotation_speed
    #chopper['rotation_speed'] = sc.abs(rotation_speed)
    #print(chopper['slit_edges'].values)
    #if rotation_speed.value < 0:
    #    chopper['slit_edges'] = sc.scalar(360.0, unit='deg') - chopper['slit_edges']
    return chopper


# TODO set L1?


def z_from_depends_on(chopper: sc.DataGroup) -> sc.Variable:
    """Compute z-component of chopper position."""
    transform = chopper['depends_on']
    return (transform * sc.vector([0, 0, 0], unit='m')).fields.z


def chopper_angular_frequency(chopper: sc.DataGroup) -> sc.Variable:
    # TODO This is a time-series log, need mechanism to use setpoint
    f = chopper['rotation_speed']
    return (2.0 * sc.Unit('rad')) * pi * f


def chopper_time_open(chopper: sc.DataGroup) -> sc.Variable:
    omega = chopper_angular_frequency(chopper)
    t = (chopper['slit_edges'][::2] + chopper['phase']) / omega
    return t.to(unit='s')


def chopper_time_close(chopper: sc.DataGroup) -> sc.Variable:
    omega = chopper_angular_frequency(chopper)
    t = (chopper['slit_edges'][1::2] + chopper['phase']) / omega
    return t.to(unit='s')


def get_frames(wfm_chopper_near: sc.DataGroup, wfm_chopper_far: sc.DataGroup,
               source: sc.DataGroup, L2: sc.Variable,
               pulse_width: sc.Variable) -> sc.Dataset:
    """
    Compute analytical frame boundaries and shifts based on chopper
    parameters and detector pixel positions.
    A set of frame boundaries is returned for each pixel.
    The frame shifts are the same for all pixels.
    See Schmakat et al. (2020);
    https://www.sciencedirect.com/science/article/pii/S0168900220308640
    for a description of the procedure.
    """
    pulse_width = pulse_width.to(unit='s')
    # Sub-frame offset: these are the mid-time point between the WFM choppers,
    # which is the same as the opening edge of the second WFM chopper in the case
    # of optically blind choppers.
    subframe_offset = chopper_time_open(wfm_chopper_far)
    #alpha = sc.to_unit(constants.m_n / constants.h, 'us/m/angstrom')

    # TODO
    # compute L2 using transform_coords
    # updating naming, paper uses z counted from source
    z_near = z_from_depends_on(wfm_chopper_near)
    z_far = z_from_depends_on(wfm_chopper_far)
    z_source = z_from_depends_on(source)

    # Distance between WFM choppers
    dz_wfm = sc.abs(z_near - z_far)
    # Mid-point between WFM choppers
    z_wfm = 0.5 * (z_near + z_far)
    # Ratio of WFM chopper distances
    z_ratio_wfm = (z_far - z_source) / (z_near - z_source)
    # Distance between detector positions and wfm chopper mid-point
    # TODO rename to Lwfm, distance from wfm to sample?
    zdet_minus_zwfm = sc.abs(z_wfm) + L2

    print(f'{dz_wfm=}')
    print(f'{z_wfm=}')
    print(f'{z_ratio_wfm=}')
    print(f'{zdet_minus_zwfm=}')

    # Find delta_t for the min and max wavelengths:
    # dt_lambda_max is equal to the time width of the WFM choppers windows
    dt_lambda_max = chopper_time_close(wfm_chopper_near) - chopper_time_open(
        wfm_chopper_near)

    print(f'{dt_lambda_max=}')

    # t_lambda_max is found from the relation between t and delta_t: equation (2) in
    # Schmakat et al. (2020).
    t_lambda_max = (dt_lambda_max / dz_wfm) * zdet_minus_zwfm
    print(t_lambda_max)

    # t_lambda_min is found from the relation between lambda_N and lambda_N+1,
    # equation (3) in Schmakat et al. (2020).
    t_lambda_min = t_lambda_max * z_ratio_wfm - pulse_width * (
        zdet_minus_zwfm / sc.abs(z_near - z_source))

    # dt_lambda_min is found from the relation between t and delta_t: equation (2)
    # in Schmakat et al. (2020), and using the expression for t_lambda_max.
    dt_lambda_min = dt_lambda_max * z_ratio_wfm - pulse_width * dz_wfm / sc.abs(
        z_near - z_source)

    # Frame edges and resolutions for each pixel.
    # The frames do not stop at t_lambda_min and t_lambda_max, they also include the
    # fuzzy areas (delta_t) at the edges.
    time_min = t_lambda_min - (0.5 * dt_lambda_min) + subframe_offset
    time_max = t_lambda_max + (0.5 * dt_lambda_max) + subframe_offset
    return {
        'subframe_start': time_min,
        'subframe_stop': time_max,
        'subframe_offset': subframe_offset
    }
