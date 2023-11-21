# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import io

import scipp as sc

from scippneutron.io import cif


# TODO
def test_asd():
    meas = sc.DataArray(
        10 * sc.arange('time', 5.0), coords={'time': sc.arange('time', 1, 6, unit='m')}
    )
    meas.variances = 0.1 * sc.arange('x', 5)
    meas.name = 'intensity'

    tof1 = sc.DataGroup({'tof_meas': meas})

    buffer = io.StringIO()
    cif.save_cif(buffer, blocks={'tof1': tof1})
    buffer.seek(0)
    print(buffer.read())
