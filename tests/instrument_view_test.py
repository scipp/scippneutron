# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import matplotlib
import numpy as np
import scipp as sc

import scippneutron as scn

matplotlib.use('Agg')


def make_detector_bank(center: list | tuple | None = None) -> sc.DataArray:
    if center is None:
        center = (0, 0, 0)

    nx, ny = 20, 12
    dx, dy = 0.1, 0.1
    x = np.arange(-nx // 2, nx // 2) * dx + center[0]
    y = np.arange(-ny // 2, ny // 2) * dy + center[1]
    z = np.zeros(ny * nx, dtype=float) + center[2]
    positions = np.meshgrid(x, y)
    positions = np.array(positions).T.reshape(-1, 2)
    positions = np.concatenate([positions, z[:, None]], axis=1)

    nt = 100
    da = sc.DataArray(
        data=sc.array(
            dims=['pixel', 'time'],
            values=np.random.uniform(0, 100, size=(positions.shape[0], nt)),
        ),
        coords={
            'position': sc.vectors(dims=['pixel'], values=positions, unit='m'),
            'time': sc.linspace('time', start=0, stop=7.1e4, num=nt + 1, unit='us'),
        },
    )
    return da


def test_instrument_view():
    bank = make_detector_bank(center=(0, 0, 5))
    scn.instrument_view(bank, size=0.1)


def test_instrument_view_two_banks_dict():
    bank1 = make_detector_bank(center=(-1.5, 0, 5))
    bank2 = make_detector_bank(center=(1.5, 0, 5))
    scn.instrument_view({"bank1": bank1, "bank2": bank2}, size=0.1)


def test_instrument_view_two_banks_datagroup():
    bank1 = make_detector_bank(center=(-1.5, 0, 5))
    bank2 = make_detector_bank(center=(1.5, 0, 5))
    scn.instrument_view(sc.DataGroup({"bank1": bank1, "bank2": bank2}), size=0.1)


def test_instrument_view_default_size():
    bank = make_detector_bank(center=(0, 0, 5))
    scn.instrument_view(bank)


def test_instrument_view_pixel_size():
    bank = make_detector_bank(center=(0, 0, 5))
    scn.instrument_view(bank, pixel_size=0.2)


def test_neutron_instrument_view_with_masks():
    bank = make_detector_bank(center=(0, 0, 5))
    bank.masks['m'] = bank.data < bank.data.max() * 0.25
    scn.instrument_view(bank)


def test_neutron_instrument_view_with_cmap_args():
    bank = make_detector_bank(center=(0, 0, 5))
    scn.instrument_view(bank, size=0.1, vmin=10.0, vmax=50.0, cmap="magma", logc=True)
