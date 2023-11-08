# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc

from scippneutron.io.table import serialize_to_table


def test_asd():
    da = sc.DataArray(-sc.arange('x', 5), coords={'x': sc.arange('x', 5, unit='m')})
    print(list(serialize_to_table(da, units=False)))
