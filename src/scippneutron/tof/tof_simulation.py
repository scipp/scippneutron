# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc

from .unwrap import (
    Choppers,
    Facility,
    NumberOfNeutrons,
    SimulationResults,
    SimulationSeed,
)


def run_tof_simulation(
    facility: Facility,
    choppers: Choppers,
    seed: SimulationSeed,
    number_of_neutrons: NumberOfNeutrons,
) -> SimulationResults:
    import tof as tof_pkg

    tof_choppers = [
        tof_pkg.Chopper(
            frequency=abs(ch.frequency),
            direction=tof_pkg.AntiClockwise
            if (ch.frequency.value > 0.0)
            else tof_pkg.Clockwise,
            open=ch.slit_begin,
            close=ch.slit_end,
            phase=abs(ch.phase),
            distance=ch.axle_position.fields.z,
            name=name,
        )
        for name, ch in choppers.items()
    ]
    source = tof_pkg.Source(facility=facility, neutrons=number_of_neutrons, seed=seed)
    if not tof_choppers:
        events = source.data.squeeze()
        return SimulationResults(
            time_of_arrival=events.coords['time'],
            speed=events.coords['speed'],
            wavelength=events.coords['wavelength'],
            weight=events.data,
            distance=0.0 * sc.units.m,
        )
    model = tof_pkg.Model(source=source, choppers=tof_choppers)
    results = model.run()
    # Find name of the furthest chopper in tof_choppers
    furthest_chopper = max(tof_choppers, key=lambda c: c.distance)
    events = results[furthest_chopper.name].data.squeeze()
    events = events[
        ~(events.masks['blocked_by_others'] | events.masks['blocked_by_me'])
    ]
    return SimulationResults(
        time_of_arrival=events.coords['toa'],
        speed=events.coords['speed'],
        wavelength=events.coords['wavelength'],
        weight=events.data,
        distance=furthest_chopper.distance,
    )
