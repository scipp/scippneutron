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
    r"""A disk chopper.

    Definitions
    -----------

    Attribute names correspond closely to the names use by NeXus' ``NXdisk_chopper``.
    See https://manual.nexusformat.org/classes/base_classes/NXdisk_chopper.html
    for an overview.

    Here is how those attributes are interpreted in ScippNeutron:
    The image below shows a disk chopper with a single slit
    as seen from the neutron source looking towards the sample.
    Note that all definitions are independent of the rotation direction.

    - *TDC* (top-dead-center sensor) corresponds to a sensor that
      tracks the rotation of the chopper.
      It serves as a reference point for defining angles.
    - :attr:`DiskChopper.beam_position` is the angle :math:`\tilde{\theta}` under which
      the beam hits the chopper.
      We do not care about the radial position and assume that it can
      pass through all chopper slits.
    - The slit is defined in terms of *begin* (:attr:`DiskChopper.slit_begin`,
      :math:`\theta` in the image) and *end* (:attr:`DiskChopper.slit_end`) angles.

    .. image:: /_static/chopper-coordinates.svg
       :width: 400
       :align: center

    Quantities relating to time are defined as:

    - The chopper rotates with a frequency of :attr:`DiskChopper.rotation_speed`
      :math:`f` which is also available as :attr:`DiskChopper.angular_frequency`
      :math:`\omega = 2\pi / f`.
      A positive frequency means anticlockwise rotation and a negative frequency
      clockwise rotation.
    - :attr:`DiskChopper.top_dead_center` stores timestamps of when the
      TDC sensor registers a full rotation.
      This serves as a reference time for the chopper.
    - The chopper time :math:`t` relates to the global time of the facility
      :math:`t_g` via :math:`t_g = t + \delta t`, where :math:`\delta t`
      is :attr:`DiskChopper.delay`.
    - There is also a :attr:`DiskChopper.phase` parameter that encodes the phase
      of the chopper relative to the neutron source.
      It is unused in time calculations as the above attributes have
      all required information.

    Slit openings
    -------------

    The terminology here differentiates slit 'begin' and 'end' from 'open' and 'close'.
    The former refer to the angles relative to TDC as shown in the image above.
    The latter refer to the opening and closing times of the slit.

    It is possible to have ``end > 360 deg`` if a slit spans TDC.

    For a given slit, we require ``begin < end``.
    To also have ``open < close`` for both directions of rotation,
    we have the following correspondence:

    - clockwise rotation: ``begin`` <-> ``open`` and ``end`` <-> ``close``
    - anticlockwise rotation: ``begin`` <-> ``close`` and ``end`` <-> ``open``

    Time calculations
    -----------------

    Given the definitions above, the time in the global timing system when a point
    at angle :math:`\theta` is at the beam position is

    .. math::

        t_g(\theta) = t_0 + \delta t + \begin{cases}
        \frac{\theta-\tilde{\theta}}{\omega}, & \textsf{clockwise}\\
        \frac{2\pi - (\theta-\tilde{\theta})}{\omega}, & \textsf{anticlockwise}
        \end{cases}

    This is implemented by :meth:`DiskChopper.time_angle_at_beam` and specifically for
    the slit edges by :meth:`DiskChopper.time_open` and
    :meth:`DiskChopper.time_close`.
    """

    position: sc.Variable
    """Position of the chopper.

    This is the center point of the chopper's axle in the face towards the source.
    See https://manual.nexusformat.org/classes/base_classes/NXdisk_chopper.html
    """
    rotation_speed: Union[sc.Variable, sc.DataArray]
    """Rotation frequency of the chopper."""
    beam_position: Optional[sc.Variable] = None
    """Angle where the beam crosses the chopper."""
    delay: Optional[Union[sc.Variable, sc.DataArray]] = None
    """Difference between global facility time and chopper time."""
    pair_separation: Optional[sc.Variable] = None
    """Disk spacing in direction of beam (for double choppers only)."""
    phase: Optional[sc.Variable] = None
    """Phase of the chopper rotation relative to the source pulses."""
    radius: Optional[sc.Variable] = None
    """Radius of the chopper."""
    ratio: Optional[sc.Variable] = None
    """Pulse reduction factor in relation to other choppers/fastest pulse."""
    slits: Optional[int] = None
    """Number of slits."""
    slit_angle: Optional[sc.Variable] = None
    """Angular opening of slits."""
    slit_edges: Optional[sc.Variable] = None
    """Edges of the slits as angles measured anticlockwise from top-dead-center.

    On init, either a 1d array of the form ``[begin_0, end_0, begin_1, end_1, ...]``
    with ``begin_i < end_i``.
    Or a 2d array of the form ``[[begin_0, end_0], [begin_1, end_1], ...]``.
    The order of slits is arbitrary.

    After init, a 2d array like the second option described above.
    The dim names depend on the input.
    """
    slit_height: Optional[sc.Variable] = None
    """Distance from chopper outer edge to bottom of slits."""
    top_dead_center: Optional[sc.Variable] = None
    """Timestamps of the top-dead-center sensor."""
    typ: DiskChopperType = DiskChopperType.single
    """Chopper type; currently, only :attr:`DiskChopperType.single` is supported."""
    wavelength_range: Optional[sc.Variable] = None
    """Low and high values of wavelength range transmitted."""
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

    @property
    def is_clockwise(self) -> bool:
        """Return True if the chopper rotates clockwise."""
        return self._clockwise

    def relative_time_open(self) -> sc.Variable:
        """Return the opening times of the chopper slits.

        If the ``beam_position`` is not set, it is assumed to be 0.

        One time can be negative for anticlockwise-rotating choppers with a slit
        across top-dead-center.

        Returns
        -------
        :
            Variable of opening times as offsets from the top-dead-center timestamps.

        Raises
        ------
        RuntimeError
            If no slits have been defined.

        See Also
        --------
        DiskChopper.time_open:
            Computes the absolute opening time in the global timing system.
        """
        if self.slit_edges is None:
            raise RuntimeError("No slits have been defined")
        if self._clockwise:
            return self.relative_time_angle_at_beam(self.slit_begin)
        return self.relative_time_angle_at_beam(self.slit_end)

    def relative_time_close(self) -> sc.Variable:
        """Return the closing times of the chopper slits.

        If the ``beam_position`` is not set, it is assumed to be 0.

        One time can be ``> 360 deg`` for clockwise-rotating choppers with a slit
        across top-dead-center.

        Returns
        -------
        :
            Variable of closing times as offsets from the top-dead-center timestamps.

        Raises
        ------
        RuntimeError
            If no slits have been defined.

        See Also
        --------
        DiskChopper.time_close:
            Computes the absolute closing time in the global timing system.
        """
        if self.slit_edges is None:
            raise RuntimeError("No slits have been defined")
        if self._clockwise:
            return self.relative_time_angle_at_beam(self.slit_end)
        return self.relative_time_angle_at_beam(self.slit_begin)

    def open_duration(self) -> sc.Variable:
        """Return how long the chopper is open for.

        Returns
        -------
        :
            Variable of opening durations.

        Raises
        ------
        RuntimeError
            If no slits have been defined.
        """
        return self.relative_time_close() - self.relative_time_open()

    def relative_time_angle_at_beam(self, angle: sc.Variable) -> sc.Variable:
        """Return the time when a position of the chopper is at the beam.

        The times are offsets of when the given angle passes by the
        beam position relative to the chopper's top-dead-center timestamps.

        If the ``beam_position`` is not set, it is assumed to be 0.

        Returns
        -------
        :
            Angles to compute times for.
            Defined anticlockwise with respect to top-dead-center.

        Raises
        ------
        RuntimeError
            If no slits have been defined.

        See Also
        --------
        DiskChopper.time_angle_at_beam:
            Computes absolute times in the global timing system.
        """
        # Ensure the correct output unit.
        angle = angle.to(unit='rad')

        if self.beam_position is not None:
            angle = angle - self.beam_position.to(unit=angle.unit, copy=False)
        if not self._clockwise:
            angle = sc.scalar(2.0, unit='rad') * sc.constants.pi - angle
        return angle / abs(self.angular_frequency)

    def time_open(self) -> sc.Variable:
        """Return the absolute opening times of the chopper slits.

        The times are absolute (date-)times in the global timing system as defined
        through ``top_dead_center`` and ``delay``.

        If the ``beam_position`` is not set, it is assumed to be 0.

        Returns
        -------
        :
            Variable of opening times with ``dtype=datetime``.

        Raises
        ------
        RuntimeError
            If ``slits`` or ``top_dead_center`` have been defined.

        See Also
        --------
        DiskChopper.relative_time_open:
            Computes the opening time relative to the chopper's top-dead-center.
        """
        return self._relative_to_absolute_time(self.relative_time_open())

    def time_close(self) -> sc.Variable:
        """Return the absolute closing times of the chopper slits.

        The times are absolute (date-)times in the global timing system as defined
        through ``top_dead_center`` and ``delay``.

        If the ``beam_position`` is not set, it is assumed to be 0.

        Returns
        -------
        :
            Variable of opening times with ``dtype=datetime``.

        Raises
        ------
        RuntimeError
            If ``slits`` or ``top_dead_center`` have been defined.

        See Also
        --------
        DiskChopper.relative_time_close:
            Computes the closing time relative to the chopper's top-dead-center.
        """
        return self._relative_to_absolute_time(self.relative_time_close())

    def time_angle_at_beam(self, angle: sc.Variable) -> sc.Variable:
        """Return the absolute time when a position of the chopper is at the beam.

        The times are absolute (date-)times in the global timing system as defined
        through ``top_dead_center`` and ``delay``.

        If the ``beam_position`` is not set, it is assumed to be 0.

        Returns
        -------
        :
            Angles to compute times for.
            Defined anticlockwise with respect to top-dead-center.

        Raises
        ------
        RuntimeError
            If ``slits`` or ``top_dead_center`` have been defined.

        See Also
        --------
        DiskChopper.relative_time_angle_at_beam:
            Computes times relative to the chopper's top-dead-center.
        """
        return self._relative_to_absolute_time(self.relative_time_angle_at_beam(angle))

    def _relative_to_absolute_time(self, relative_time: sc.Variable) -> sc.Variable:
        if self.top_dead_center is None:
            raise RuntimeError("No top dead center has been defined")
        res = (
            relative_time.to(unit=self.top_dead_center.unit, dtype=int, copy=False)
            + self.top_dead_center
        )
        if self.delay is not None:
            res += self.delay.to(unit=res.unit, copy=False)
        return res

    def __eq__(self, other: Any) -> Union[bool, NotImplemented]:
        if not isinstance(other, DiskChopper):
            return NotImplemented
        return all(
            _field_eq(getattr(self, field.name), getattr(other, field.name))
            for field in dataclasses.fields(self)
        )

    def make_svg(self, image_size: int = 400) -> str:
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

    def _repr_svg_(self) -> str:
        return self.make_svg()

    def _repr_html_(self) -> str:
        from .._html_repr import disk_chopper_html_repr

        return disk_chopper_html_repr(self)


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
    if edges.ndim == 1:
        edge_dim = 'edge' if edges.dim != 'edge' else 'edge_dim'
        folded = edges.fold(edges.dim, sizes={edges.dim: -1, edge_dim: 2})
        if sc.any(folded[edge_dim, 0] > folded[edge_dim, 1]):
            raise ValueError(
                "Invalid slit edges, must be given as "
                "[begin_0, end_0, begin_1, end_1, ...] where begin_n < end_n"
            )
        return folded
    if edges.ndim == 2:
        if edges.shape[1] != 2:
            raise sc.DimensionError(
                "The second dim of the slit edges must be length 2."
            )
        return edges
    else:
        raise sc.DimensionError("The slit edges must be 1- or 2-dimensional")


def _require_frequency(name: str, x: sc.Variable) -> None:
    try:
        sc.scalar(0.0, unit=x.unit).to(unit='Hz')
    except sc.UnitError:
        raise sc.UnitError(f"'{name}' must be a frequency, got unit {x.unit}")


def _len_or_1(x: sc.Variable) -> int:
    if x.ndim == 0:
        return 1
    return len(x)
