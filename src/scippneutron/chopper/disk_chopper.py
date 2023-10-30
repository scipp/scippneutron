# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import dataclasses
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


@dataclasses.dataclass(frozen=True, eq=False)
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

    position: sc.Variable
    rotation_speed: Union[sc.Variable, sc.DataArray]
    delay: Optional[Union[sc.Variable, sc.DataArray]] = None
    phase: Optional[sc.Variable] = None
    radius: Optional[sc.Variable] = None
    slits: Optional[int] = None
    slit_height: Optional[sc.Variable] = None

    slit_edges: Optional[sc.Variable] = None
    """Edges of the slits as angles measured anticlockwise from top-dead-center.

    On init, a 1d array of the form ``[begin_0, end_0, begin_1, end_1, ...]`` with
    ``begin_n < end_n``.

    After init, a 2d array of shape ``[slit, edge]`` where ``slit`` indexes the chopper
    slits and ``edge`` is of length 2 where ``edge=0`` is the beginning edge and
    ``edge=1`` the ending edge of the slit.
    The dim names depend on the input.

    Here, the 'beginning' edge is the edge of a slit with the smaller angle and the
    'ending' edge is the other.
    That is, walking around the chopper disk from top-dead-center in anticlockwise
    direction, one encounters the beginning edge before the closing edge.
    This differs from the 'opening' and 'closing' times which depend on the
    direction of rotation.
    """

    beam_position: Optional[sc.Variable] = None
    name: str = ''
    typ: DiskChopperType = DiskChopperType.single
    _clockwise: bool = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'typ', DiskChopperType(self.typ))
        if self.typ != DiskChopperType.single:
            raise NotImplementedError(
                "Only single disk choppers are supported, got " f"typ={self.typ}"
            )

        _require_frequency('rotation_speed', self.rotation_speed)
        object.__setattr__(self, 'slit_edges', _parse_slit_edges(self.slit_edges))
        if self.slits is None and self.slit_edges is not None:
            object.__setattr__(self, 'slits', self.slit_edges.shape[0])
        object.__setattr__(
            self,
            '_clockwise',
            sc.all(
                self.rotation_speed < sc.scalar(0.0, unit=self.rotation_speed.unit)
            ).value,
        )

    @property
    def slit_begin(self) -> Optional[sc.Variable]:
        """Beginning edges of the slits."""
        if self.slit_edges is None:
            return None
        return self.slit_edges[self.slit_edges.dims[1], 0]

    @property
    def slit_end(self) -> Optional[sc.Variable]:
        """Ending edges of the slits."""
        if self.slit_edges is None:
            return None
        return self.slit_edges[self.slit_edges.dims[1], 1]

    @property
    def angular_frequency(self) -> sc.Variable:
        """Rotation speed as an angular frequency in ``rad * rotation_speed.unit``."""
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

    def to_svg(self, image_size: int = 400) -> str:
        """Generate an SVG image for this chopper.

        Parameters
        ----------
        image_size:
            The size in pixels of the image.

        Returns
        -------
        :
            The SVG image as a string.
        """
        from ._svg import draw_disk_chopper

        return draw_disk_chopper(self, image_size=image_size)

    def __eq__(self, other: Any) -> Union[bool, NotImplemented]:
        if not isinstance(other, DiskChopper):
            return NotImplemented
        return all(
            _field_eq(getattr(self, field.name), getattr(other, field.name))
            for field in dataclasses.fields(self)
        )


def _field_eq(a: Any, b: Any) -> bool:
    if isinstance(a, (sc.Variable, sc.DataArray)):
        try:
            return sc.identical(a, b)
        except TypeError:
            return False  # if identical does not support b
    return a == b


def _parse_typ(typ: Union[DiskChopperType, str]) -> DiskChopperType:
    # Special shorthand for convenience
    if typ == "single":
        return DiskChopperType.single
    return DiskChopperType(typ)


def _parse_slit_edges(edges: Optional[sc.Variable]) -> Optional[sc.Variable]:
    if edges is None:
        return None
    if edges.ndim != 1:
        raise sc.DimensionError("The slit edges must be 1-dimensional")
    edge_dim = 'edge' if edges.dim != 'edge' else 'edge_dim'
    folded = edges.fold(edges.dim, sizes={edges.dim: -1, edge_dim: 2})
    if sc.any(folded[edge_dim, 0] > folded[edge_dim, 1]):
        raise ValueError(
            "Invalid slit edges, must be given as "
            "[begin_0, end_0, begin_1, end_1, ...] where begin_n < end_n"
        )
    return folded


def _require_frequency(name: str, x: sc.Variable) -> None:
    try:
        sc.scalar(0.0, unit=x.unit).to(unit='Hz')
    except sc.UnitError:
        raise sc.UnitError(f"'{name}' must be a frequency, got unit {x.unit}")


def _len_or_1(x: sc.Variable) -> int:
    if x.ndim == 0:
        return 1
    return len(x)
