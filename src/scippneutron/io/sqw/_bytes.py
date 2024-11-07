# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import enum
import sys
from typing import Literal


class Byteorder(enum.Enum):
    little = "little"
    big = "big"

    @classmethod
    def parse(cls, value: Byteorder | Literal["native", "little", "big"]) -> Byteorder:
        if isinstance(value, Byteorder):
            return value
        if isinstance(value, str):
            if value == "native":
                return cls.native()
            return cls(value)
        raise ValueError(f"Invalid Byteorder: {value}")

    @classmethod
    def native(cls) -> Byteorder:
        return cls(sys.byteorder)

    def get(self) -> Literal["little", "big"]:
        match self:
            case Byteorder.little:
                return "little"
            case Byteorder.big:
                return "big"
