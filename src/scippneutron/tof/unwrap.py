# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
"""
from typing import NewType

import scipp as sc

# SourceChopper = NewType('SourceChopper', chopper_cascade.Chopper)
#
# SourcePulse = NewType('SourcePulse', sc.Variable)
#
# FrameBounds = NewType('FrameBounds', sc.DataGroup)
# SubframeBounds = NewType('SubframeBounds', sc.DataGroup)
# FramePeriod = NewType('FramePeriod', sc.Variable)
WrappedTimeOffset = NewType('WrappedTimeOffset', sc.Variable)
TimeOffset = NewType('TimeOffset', sc.Variable)


# Should this be a helper, or a provider?
def time_offset(
    *,
    wrapped_time_offset: sc.Variable,
    time_offset_min: sc.Variable,
    frame_period: sc.Variable,
) -> sc.Variable:
    """
    Time offset from the start of the frame emitting the neutron.

    This is not identical to the time-of-flight, since the time-of-flight is measured,
    e.g., from the center of the pulse, to the center of a pulse-shaping chopper slit.

    Parameters
    ----------
    wrapped_time_offset :
        Time offset from the time-zero as recorded by the data acquisition system.
    time_offset_min :
        Minimum arrival time offset of neutrons that can pass through the chopper
        cascade. Typically pixel-dependent.
    frame_period :
        Time between the start of two consecutive frames, i.e., the period of the
        time-zero used by the data acquisition system.
    """
    wrapped_time_min = time_offset_min % frame_period
    delta = frame_period if wrapped_time_offset < wrapped_time_min else 0
    offset_frames = time_offset_min - wrapped_time_min + delta
    return offset_frames + wrapped_time_offset


# Don't use chopper, just open and close times
# We can handle WFM by adding an intermediate binning step that cuts into subframes,
# then use a subframe-dependent time_open and time_close here.
# - set source_position to chopper's position
def time_of_flight(
    *,
    time_offset: sc.Variable,
    source_time_open: sc.Variable,
    source_time_close: sc.Variable,
) -> sc.Variable:
    """
    Time-of-flight of neutrons passing through a chopper cascade.

    A chopper is used to define (1) the "source" location and (2) the time-of-flight
    time origin. The time-of-flight is then the time difference between the time of
    arrival of the neutron at the detector, and the time of arrival of the neutron at
    the chopper. L1 needs to be redefined to be the distance between the chopper and
    the sample.

    If there is no pulse-shaping chopper, then the source-pulse begin and end time
    should be set as the source_time_open and source_time_close, respectively.

    For WFM, the source_time_open and source_time_close will be different for each
    subframe. In this case, all input parameters should be given as variables with
    subframe as dimension.

    Parameters
    ----------
    time_offset :
        Time offset from the start of the frame emitting the neutron.
    source_time_open :
        Time at which the source chopper opens.
    source_time_close :
        Time at which the source chopper closes.
    """
    # TODO Need to handle choppers with multiple openings, where we need to select one
    time_zero = 0.5 * (source_time_open + source_time_close)
    return time_offset - time_zero


def compute_time_of_flight(da: sc.DataArray, choppers) -> sc.DataArray:
    # Outline:
    # 1. Use chopper_cascade module to compute time_offset_min
    # 2. Compute time_offset
    # 3. Compute time_of_flight, based on "source chopper" openings
    #
    # In case of WFM:
    # 1. Use chopper_cascade module to compute time_offset_min and subframe_time_bounds
    # 2. Compute time_offset
    # 3. Use scipp.bin to bin into subframes
    # 4. Compute time_of_flight
    # 5. Concat subframes, based on "source chopper" openings. The source chopper is
    #    the WFM chopper, or a combination of WFM choppers.
    raise NotImplementedError
