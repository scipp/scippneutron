# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

from typing import Any, Optional, Union

import scipp as sc
import scipp.constants

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
        phase: Optional[sc.Variable] = None,
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
        self._phase = phase
        self._radius = radius
        self._slits = slits
        self._slit_height = slit_height
        self._slit_edges = _parse_slit_edges(slit_edges)

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
    def phase(self) -> Optional[sc.Variable]:
        return self._phase

    @property
    def radius(self) -> Optional[sc.Variable]:
        return self._radius

    @property
    def slits(self) -> Optional[int]:
        if self._slits is None:
            if self.slit_edges is not None:
                self._slits = self.slit_edges.shape[0]
        return self._slits

    @property
    def slit_height(self) -> Optional[sc.Variable]:
        return self._slit_height

    @property
    def slit_edges(self) -> Optional[sc.Variable]:
        """The beginning and end edges of each slit.

        Returns
        -------
        :
            Variable of shape ``[slit, edge]`` where ``slit`` indexes the chopper slits
            and ``edge`` is of length 2 where ``edge=0`` is the beginning edge and
            ``edge=1`` the end edge of the slit.
            The dim names depend on the input.
        """
        return self._slit_edges

    @property
    def slit_begin(self) -> Optional[sc.Variable]:
        if self.slit_edges is None:
            return None
        return self.slit_edges[self.slit_edges.dims[1], 0]

    @property
    def slit_end(self) -> Optional[sc.Variable]:
        if self.slit_edges is None:
            return None
        return self.slit_edges[self.slit_edges.dims[1], 1]

    @property
    def angular_frequency(self) -> sc.Variable:
        return sc.scalar(2.0, unit="rad") * sc.constants.pi * self.rotation_speed

    def time_open(self) -> sc.Variable:
        """Return the times when chopper windows open.

        These are time offsets of when the slit start edges pass by the
        top dead center relative to the start of a cycle.

        Returns
        -------
        :
            Variable of opening times.

        Raises
        ------
        RuntimeError
            If no slits have been defined.
        """
        if self.slit_edges is None:
            raise RuntimeError("No slits have been defined")
        return self._time_open_close(self.slit_begin)

    def time_close(self) -> sc.Variable:
        """Return the times when chopper windows close.

        These are time offsets of when the slit end edges pass by the
        top dead center relative to the start of a cycle.

        Returns
        -------
        :
            Variable of opening times.

        Raises
        ------
        RuntimeError
            If no slits have been defined.
        """
        if self.slit_edges is None:
            raise RuntimeError("No slits have been defined")
        return self._time_open_close(self.slit_end)

    def _time_open_close(self, edge: sc.Variable) -> sc.Variable:
        if self.phase is not None:
            edge = edge + self.phase.to(unit=edge.unit, copy=False)
        return sc.to_unit(
            edge / self.angular_frequency,
            sc.reciprocal(self.rotation_speed.unit),
            copy=False,
        )

    def open_duration(self) -> sc.Variable:
        """Return the lengths of the open windows of the chopper.

        Returns
        -------
        :
            Variable of opening durations.

        Raises
        ------
        RuntimeError
            If no slits have been defined.
        """
        return self.time_close() - self.time_open()

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
        regular = (
            'typ',
            'name',
            'position',
            'rotation_speed',
            'phase',
            'slit_edges',
            'slit_height',
            'radius',
        )
        computed = ('slits',)
        return all(
            eq(getattr(self, key), getattr(other, key)) for key in regular
        ) and all(computed_eq(key) for key in computed)


def _parse_typ(typ: Union[DiskChopperType, str]) -> DiskChopperType:
    # Special shorthand for convenience
    if typ == "single":
        return DiskChopperType.single
    return DiskChopperType(typ)


def _parse_slit_edges(edges: Optional[sc.Variable]) -> Optional[sc.Variable]:
    if edges is None:
        return None
    if edges.ndim == 1:
        edge_dim = 'edge' if edges.dim != 'edge' else 'edge_dim'
        return edges.fold(edges.dim, sizes={edges.dim: -1, edge_dim: 2})
    raise sc.DimensionError("The slit edges must be 1-dimensional")


def _require_frequency(name: str, x: sc.Variable) -> None:
    try:
        sc.scalar(0.0, unit=x.unit).to(unit='Hz')
    except sc.UnitError:
        raise sc.UnitError(f"'{name}' must be a frequency, got unit {x.unit}")


def _len_or_1(x: sc.Variable) -> int:
    if x.ndim == 0:
        return 1
    return len(x)
