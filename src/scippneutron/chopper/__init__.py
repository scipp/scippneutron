# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""Chopper utilities."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        'disk_chopper': ['DiskChopper', 'DiskChopperType'],
        'filtering': ['collapse_plateaus', 'filter_in_phase', 'find_plateaus'],
        'nexus_chopper': ['extract_chopper_from_nexus'],
    },
)
