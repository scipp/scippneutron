import scipp as sc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class Beamline:
    def __init__(self,
                 ax,
                 *,
                 tmax,
                 Lmax,
                 frame_rate=14.0 * sc.Unit('Hz'),
                 frame_offset=0.0 * sc.Unit('ms')):
        self._ax = ax
        self._time_unit = sc.Unit('ms')
        self._frame_length = (1.0 / frame_rate).to(unit=self._time_unit)
        self._tmax = tmax.to(unit=self._time_unit)
        self._Lmax = Lmax.to(unit='m')
        self._frame_offset = frame_offset.to(unit=self._time_unit)
        self._ax.set_xlabel(f"Time [{self._time_unit}]")
        self._ax.set_ylabel("Distance [m]")

    def add_annotations(self):
        props = dict(arrowstyle='-|>')
        x0 = self._frame_length
        x1 = self._frame_length + self._frame_offset
        self._ax.annotate(r'$T_0^i$',
                          xy=(x0.value, 0),
                          xytext=(x0.value - 10, 5),
                          arrowprops=props)
        self._ax.annotate(r'$T_0^i+\Delta T_0$',
                          xy=(x1.value, 0),
                          xytext=(x1.value + 3, 3),
                          arrowprops=props)

    def add_source_pulse(self, pulse_length=3.0 * sc.Unit('ms')):
        self._pulse_length = pulse_length.to(unit=self._time_unit)
        # Define and draw source pulse
        x0 = 0.0
        x1 = self._pulse_length.value
        y0 = 0.0
        psize = 1.0
        rect = Rectangle((x0, y0),
                         x1,
                         -psize,
                         lw=1,
                         fc='orange',
                         ec='k',
                         hatch="////",
                         zorder=10)
        self._ax.add_patch(rect)
        x0 += self._frame_length.value
        rect = Rectangle((x0, y0),
                         x1,
                         -psize,
                         lw=1,
                         fc='orange',
                         ec='k',
                         hatch="////",
                         zorder=10)
        self._ax.add_patch(rect)
        self._ax.text(x0,
                      -psize,
                      f"Source pulse ({pulse_length.value} {pulse_length.unit})",
                      ha="left",
                      va="top",
                      fontsize=6)

    def add_event_time_zero(self):
        ls = 'dotted'
        x = 0
        while x < self._tmax.value:
            self._ax.axvline(x=x, ls=ls)
            x += self._frame_length.value

    def add_neutron_pulse(self, tof_min=160.0 * sc.Unit('ms')):
        x0 = self._frame_offset
        x1 = x0  # + self._pulse_length
        x3 = x1 + tof_min + 0.95 * self._frame_length  # small gap
        x4 = x0 + tof_min
        y0 = 0
        y1 = self._Lmax.value
        x = sc.concat([x0, x1, x3, x4], 'x')
        self._ax.fill(x.values, [y0, y0, y1, y1], alpha=0.3)
        x += self._frame_length
        self._ax.fill(x.values, [y0, y0, y1, y1], alpha=0.3)

    def add_detector(self, *, distance, name='detector'):
        # TODO This could accept a list of positions and plot a rectangle from min to
        # max detector distance
        self._ax.plot([0, self._tmax.max().value], [distance.value, distance.value],
                      lw=3,
                      color='grey')
        self._ax.text(0.0, distance.value, name, va="bottom", ha="left")

    def add_sample(self, *, distance):
        self._ax.plot([0, self._tmax.max().value], [distance.value, distance.value],
                      lw=3,
                      color='green')
        self._ax.text(0.0, distance.value, 'sample', va="bottom", ha="left")


def time_distance_diagram(tmax=300 * sc.Unit('ms')):
    fig, ax = plt.subplots(1, 1)
    beamline = Beamline(ax,
                        tmax=tmax,
                        Lmax=40.0 * sc.Unit('m'),
                        frame_offset=1.5 * sc.Unit('ms'))
    beamline.add_event_time_zero()
    beamline.add_source_pulse()
    beamline.add_sample(distance=20.0 * sc.Unit('m'))
    det1 = 30.0 * sc.Unit('m')
    det2 = 40.0 * sc.Unit('m')
    beamline.add_detector(distance=det1, name='detector1')
    beamline.add_detector(distance=det2, name='detector2')
    beamline.add_neutron_pulse()
    beamline.add_annotations()
    props = dict(arrowstyle='-|>')
    ax.annotate(r'$T_0^{i+1}+t_{\mathrm{pivot}}(\mathrm{det1})$',
                xy=(191, det1.value),
                xytext=(150, 10),
                arrowprops=props)
    ax.annotate(r'$T_0^{i+2}+t_{\mathrm{pivot}}(\mathrm{det2})$',
                xy=(230, det2.value),
                xytext=(220, 15),
                arrowprops=props)

    return fig
