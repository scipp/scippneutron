# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)


from typing import Union

import scipp as sc

try:
    # Python 3.11+
    from enum import StrEnum

    class DiskChopperType(StrEnum):
        """Type of disk chopper."""

        single = 'Chopper type single'
        contra_rotating_pair = 'contra_rotating_pair'
        synchro_pair = 'synchro_pair'

    del StrEnum

except ImportError:
    from enum import Enum

    class DiskChopperType(str, Enum):  # type: ignore[no-redef]
        """Type of disk chopper."""

        single = 'Chopper type single'
        contra_rotating_pair = 'contra_rotating_pair'
        synchro_pair = 'synchro_pair'

    del Enum


class DiskChopper:
    def __init__(
        self,
        *,
        typ: Union[DiskChopperType, str],
        position: sc.Variable,
        rotation_speed: sc.Variable,
        name: str = '',
    ) -> None:
        self._typ = _parse_typ(typ)
        self._position = position
        self._rotation_speed = rotation_speed
        self._name = name

    @property
    def typ(self) -> DiskChopperType:
        return self._typ

    @property
    def position(self) -> sc.Variable:
        return self._position

    @property
    def rotation_speed(self) -> sc.Variable:
        return self._rotation_speed

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return (
            f"DiskChopper(typ={self.typ}, name={self.name}, "
            f"position={self.position})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DiskChopper):
            return NotImplemented
        return (
            self.typ == other.typ
            and sc.identical(self.position, other.position)
            and sc.identical(self.rotation_speed, other.rotation_speed)
            and self.name == other.name
        )


def _parse_typ(typ: Union[DiskChopperType, str]) -> DiskChopperType:
    # Special shorthand for convenience
    if typ == "single":
        return DiskChopperType.single
    return DiskChopperType(typ)
