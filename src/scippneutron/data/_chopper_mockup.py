# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc


def chopper_mockup() -> sc.DataGroup:
    """Return a data group for a fake chopper.

    Returns
    -------
    :
        Data that looks similar to a chopper that was loaded
        from a NeXus ``NXdisk_chopper``.
    """
    rotation_speed = _rotation_speed()
    return sc.DataGroup(
        {
            'delay': sc.DataGroup(
                {
                    'value': sc.DataArray(
                        sc.array(dims=['time'], values=[3050], unit='ns'),
                        coords={
                            'time': sc.datetimes(
                                dims=['time'], values=['2023-01-19T08:11:06'], unit='ns'
                            )
                        },
                    ),
                }
            ),
            'position': sc.vector([0.0, 0.0, 60.0], unit='m'),
            'radius': sc.scalar(0.35, unit='m'),
            'rotation_speed': sc.DataGroup(
                {
                    'value': rotation_speed,
                }
            ),
            'slit_height': sc.scalar(0.1, unit='m'),
            'slit_edges': sc.array(
                dims=['slit'], values=[30.0, 160.0, 210.0, 280.0], unit='deg'
            ),
            'slits': 2,
            'top_dead_center': sc.DataGroup(
                {
                    'time': _top_dead_center(rotation_speed),
                }
            ),
        }
    )


def _rotation_speed(target_frequency: float = 14.0) -> sc.DataArray:
    time = _time_coord()

    # Arbitrary time scale.
    t = sc.linspace('time', 0, 96, 100, unit='s')
    omega = sc.linspace('time', 0.8, 0.0, 100, unit='rad/s')
    lam = sc.scalar(0.3, unit='1/s')
    phi = sc.scalar(0.9, unit='rad')
    scale = sc.scalar(0.1)
    f_plateau = target_frequency + scale * sc.sin(omega * t + phi) * sc.exp(-lam * t)
    f_plateau.unit = 'Hz'

    shoulder = (100 - np.tanh(0.7 * (t.values - 80))) / 101
    f_plateau.values *= shoulder

    f_rising = sc.linspace('time', 0.0, target_frequency, 40, unit='Hz')
    f_falling = sc.linspace('time', f_plateau[-1].value, 0.0, 40, unit='Hz')
    f = sc.concat([f_rising, f_plateau, f_falling], dim='time')

    rng = np.random.default_rng(84391)
    noise = sc.array(dims=['time'], values=rng.normal(0.0, 4e-5, 180), unit='Hz')

    return sc.DataArray(f + noise, coords={'time': time})


def _top_dead_center(rotation_speed: sc.DataArray) -> sc.Variable:
    """Compute TDC timestamps for a given rotation speed measurement.

    This assumes that the rotation speed was measured at a coarser rate than TDC and
    that the reported speed values are the speed at that moment.
    The function uses a linear interpolation between speed measurements to determine
    the momentary rotation speed.
    """
    # This function first picks a time value τ near the first measured
    # point of the rotation speed.
    # Then, τ is stepped forwards according to the estimated rotation speed at τ,
    # see the docstring about interpolation.
    #
    # Given two neighboring TDC timestamps τ_{i-1} < τ_i, we have
    #     τ_i
    #      ∫ f(t) dt = 1
    #   τ_{i-1}
    # where f(t) is the rotation speed obtained by interpolation.
    # Solving this equation gives a function τ_i(τ_{i-1}) for computing a timestamp
    # τ_i from the previous τ_{i-1}.

    time = rotation_speed.coords['time'] - sc.epoch(
        unit=rotation_speed.coords['time'].unit
    )
    slope, offset = _interpolate_rotation_speed(speed=rotation_speed.data, time=time)

    tdc = []
    # Arbitrary shift to mis-align tdc and speed timestamps.
    tau = time[0].copy().to(dtype='float64') + sc.scalar(12_000_000, unit='ns')
    while tau < time[-1]:
        tdc.append(tau)
        m = slope[tau]
        d = offset[tau]
        if m.value == 0.0:
            tau = tau + sc.reciprocal(d).to(unit=tau.unit)
        else:
            f = sc.to_unit(m * tau, d.unit) + d
            tau = (sc.sqrt(f**2 + 2 * m) - d) / m
            tau = tau.to(unit=time.unit)

    return sc.concat(tdc, dim='time').to(unit='ns', dtype='int64') + sc.epoch(unit='ns')


def _interpolate_rotation_speed(*, speed: sc.Variable, time: sc.Variable):
    """Interpolate the speed-vs-time curve with line segments."""
    m = sc.to_unit((speed[1:] - speed[:-1]) / (time[1:] - time[:-1]), 'Hz^2')
    d = speed[:-1] - sc.to_unit(m * time[:-1], speed.unit)

    slope = sc.lookup(sc.DataArray(m, coords={'time': time.to(dtype='float64')}))
    offset = sc.lookup(sc.DataArray(d, coords={'time': time.to(dtype='float64')}))
    return slope, offset


def _time_coord() -> sc.Variable:
    dt_min = 1674115866205830382
    dt_max = 1674116082246890859
    dt = sc.linspace('time', dt_min, dt_max + 1, 180, dtype='int64', unit='ns')
    return sc.epoch(unit='ns') + dt
