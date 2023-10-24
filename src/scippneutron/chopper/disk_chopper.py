# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

from typing import Any, Optional, Union

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
    """A disk chopper.

    Attribute names correspond closely to the names use by NeXus' ``NXdisk_chopper``.
    See https://manual.nexusformat.org/classes/base_classes/NXdisk_chopper.html
    for an overview.

    This class can transparently compute some quantities from others.
    So not all attributes need to be given to construct a chopper instance.
    However, it does **not** perform any consistency checks.
    If a quantity is given, it is used directly instead of computing it from related
    quantities.
    """

    def __init__(
        self,
        *,
        typ: Union[DiskChopperType, str],
        position: sc.Variable,
        rotation_speed: Union[sc.Variable, sc.DataArray],
        name: str = '',
        delay: Optional[Union[sc.Variable, sc.DataArray]] = None,
        radius: Optional[sc.Variable] = None,
        slits: Optional[int] = None,
        slit_height: Optional[sc.Variable] = None,
        slit_edges: Optional[sc.Variable] = None,
    ) -> None:
        _require_frequency('rotation_speed', rotation_speed)

        self._typ = _parse_typ(typ)
        self._position = position
        self._rotation_speed = rotation_speed
        self._name = name

        self._delay = delay
        self._radius = radius
        self._slits = slits
        self._slit_height = slit_height
        self._slit_edges = slit_edges

    @property
    def typ(self) -> DiskChopperType:
        return self._typ

    @property
    def position(self) -> sc.Variable:
        return self._position

    @property
    def rotation_speed(self) -> Union[sc.Variable, sc.DataArray]:
        return self._rotation_speed

    @property
    def name(self) -> str:
        return self._name

    @property
    def delay(self) -> Optional[Union[sc.Variable, sc.DataArray]]:
        return self._delay

    @property
    def radius(self) -> Optional[sc.Variable]:
        return self._radius

    @property
    def slits(self) -> Optional[int]:
        if self._slits is None:
            if self._slit_height is not None:
                self._slits = _len_or_1(self._slit_height)
            elif self.slit_edges is not None:
                self._slits = len(self._slit_edges) // 2
        return self._slits

    @property
    def slit_height(self) -> Optional[sc.Variable]:
        return self._slit_height

    @property
    def slit_edges(self) -> Optional[sc.Variable]:
        return self._slit_edges

    def __repr__(self) -> str:
        return (
            f"DiskChopper(typ={self.typ}, name={self.name}, "
            f"position={self.position})"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DiskChopper):
            return NotImplemented

        def eq(a: Any, b: Any) -> bool:
            try:
                return sc.identical(a, b)
            except TypeError:
                return a == b

        def computed_eq(key: str) -> bool:
            a, b = getattr(self, '_' + key), getattr(other, '_' + key)
            if (a is None) ^ (b is None):
                return eq(getattr(self, key), getattr(other, key))
            # Avoid computing it if not needed.
            return eq(a, b)

        # TODO add missing fields
        regular = ('typ', 'name', 'position', 'rotation_speed')
        computed = ('slits', 'slit_edges', 'slit_height', 'radius')
        return all(
            eq(getattr(self, key), getattr(other, key)) for key in regular
        ) and all(computed_eq(key) for key in computed)


def _parse_typ(typ: Union[DiskChopperType, str]) -> DiskChopperType:
    # Special shorthand for convenience
    if typ == "single":
        return DiskChopperType.single
    return DiskChopperType(typ)


def _require_frequency(name: str, x: sc.Variable) -> None:
    try:
        sc.scalar(0.0, unit=x.unit).to(unit='Hz')
    except sc.UnitError:
        raise sc.UnitError(f"'{name}' must be a frequency, got unit {x.unit}")


def _len_or_1(x: sc.Variable) -> int:
    if x.ndim == 0:
        return 1
    return len(x)
