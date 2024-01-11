# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import scipp as sc
from matplotlib.patches import Rectangle
from scipp.constants import h, m_n

from .._utils import as_float_type, elem_unit


def _tof_from_wavelength(
    *, wavelength: sc.Variable, Ltotal: sc.Variable
) -> sc.Variable:
    scale = (m_n / h).to(unit=sc.units.us / elem_unit(Ltotal) / elem_unit(wavelength))
    return as_float_type(Ltotal * scale, wavelength) * wavelength


class TimeDistanceDiagram:
    def __init__(self, ax, *, tmax, frame_rate=None):
        self._ax = ax
        self._time_unit = sc.Unit('ms')
        self._distance_unit = sc.Unit('m')
        frame_rate = 14.0 * sc.Unit('Hz') if frame_rate is None else frame_rate
        self._frame_length = (1.0 / frame_rate).to(unit=self._time_unit)
        self._tmax = tmax.to(unit=self._time_unit)
        self._ax.set_xlabel(f"time [{self._time_unit}]")
        self._ax.set_ylabel("distance [m]")

    @property
    def frame_length(self):
        return self._frame_length

    def to_time(self, time):
        return time.to(unit=self._time_unit)

    def to_distance(self, distance):
        return distance.to(unit=self._distance_unit)

    def annotate(self, text, *, xy, xytext, **kwargs):
        def to_mpl(point):
            x, y = point
            return self.to_time(x).value, self.to_distance(y).value

        self._ax.annotate(text, xy=to_mpl(xy), xytext=to_mpl(xytext), **kwargs)

    def add_source_pulse(self, pulse_length=None, ls='dotted'):
        pulse_length = 3.0 * sc.Unit('ms') if pulse_length is None else pulse_length
        t0 = 0.0
        t1 = self.to_time(pulse_length).value
        self._ax.text(
            t0,
            -2,
            f"Source pulse ({pulse_length.value} {pulse_length.unit})",
            ha="left",
            va="top",
            fontsize=6,
        )
        while t0 < self._tmax.value:
            rect = Rectangle((t0, 0), t1, -1, lw=1, fc='orange', ec='k')
            self._ax.add_patch(rect)
            self._ax.axvline(x=t0, ls=ls)
            t0 += self._frame_length.value

    def add_neutron(
        self,
        *,
        time_offset: sc.Variable,
        wavelength: sc.Variable,
        L: sc.Variable,
        label=None,
        color='black',
        ls='solid',
        lw=0.7,
    ):
        tof = self.to_time(_tof_from_wavelength(wavelength=wavelength, Ltotal=L))
        t0 = self.to_time(time_offset).value
        self._ax.plot(
            [t0, t0 + tof.value], [0, L.value], marker='', color=color, ls=ls, lw=lw
        )
        if label is not None:
            self._ax.text(tof.value, L.value, label, ha="center", va="bottom")

    def add_neutrons(
        self,
        *,
        lambda_min: sc.Variable,
        lambda_max: sc.Variable = None,
        Lmin: sc.Variable = 0.0 * sc.units.m,
        Lmax: sc.Variable,
        time_offset: sc.Variable,
        stride=1,
        frames=2,
    ):
        """
        Draw a wavelength band to depict propagation of neutrons. Neutrons are assumed
        to be emitted from a single point, i.e., no resolution effects are taken into
        account.

        Parameters
        ----------
        Lmin:
            Distance where neutrons are "emitted", such as the source pulse or
            a chopper. The default is at 0.0, i.e., the source position.
        Lmax:
            Distance where propagation of neutrons stops. This is typically set
            to the last detector (or after), but could be set to a chopper distance if
            a chopper extracts a smaller wavelength band.
        lambda_min:
            Minimum wavelength, defining fastest neutrons.
        lambda_max:
            Maximum wavelength, defining slowest neutrons. If lambda_max
            is None (the default) it is set such that there is no frame overlap at Lmax.
        time_offset:
            Offset time at which neutrons are emitted.
        frames:
            The number of frames that should be drawn.
        """
        tof_min = self.to_time(
            _tof_from_wavelength(wavelength=lambda_min, Ltotal=Lmax - Lmin)
        )
        if lambda_max is None:
            tof_max = tof_min + (stride - 1 + 0.95) * self._frame_length  # small 5% gap
        else:
            tof_max = self.to_time(
                _tof_from_wavelength(wavelength=lambda_max, Ltotal=Lmax - Lmin)
            )
        time_offset = self.to_time(time_offset)
        t0 = time_offset
        tmin = t0 + tof_min
        tmax = t0 + tof_max
        t = sc.concat([t0, t0, tmax, tmin], 't')
        L = sc.concat([Lmin, Lmin, Lmax, Lmax], 'L')
        for _ in range(frames):
            self._ax.fill(t.values, L.values, alpha=0.3)
            t += stride * self._frame_length

    def add_detector(self, *, distance, name='detector'):
        # TODO This could accept a list of positions and plot a rectangle from min to
        # max detector distance
        self._ax.plot(
            [0, self._tmax.max().value],
            [distance.value, distance.value],
            lw=3,
            color='grey',
        )
        self._ax.text(0.0, distance.value, name, va="bottom", ha="left")

    def add_sample(self, *, distance):
        self._ax.plot(
            [0, self._tmax.max().value],
            [distance.value, distance.value],
            lw=3,
            color='green',
        )
        self._ax.text(0.0, distance.value, 'sample', va="bottom", ha="left")
