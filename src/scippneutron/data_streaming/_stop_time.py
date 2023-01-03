# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass


@dataclass(frozen=True)
class StopTimeUpdate:
    stop_time_ms: int  # milliseconds from unix epoch
