# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import matplotlib.pyplot as plt
import scipp as sc

from scippneutron.tof import TimeDistanceDiagram


def default_frame_diagram(tmax=None):
    fig, ax = plt.subplots(1, 1)
    diagram = TimeDistanceDiagram(
        ax, tmax=300 * sc.Unit('ms') if tmax is None else tmax
    )
    det1 = 32.0 * sc.Unit('m')
    det2 = 40.0 * sc.Unit('m')
    lambda_min = 16.0 * sc.units.angstrom
    frame_offset = diagram.to_time(6.0 * sc.Unit('ms'))
    diagram.add_source_pulse(15 * sc.Unit('ms'))
    diagram.add_sample(distance=20.0 * sc.Unit('m'))
    diagram.add_detector(distance=det1, name='detector1')
    diagram.add_detector(distance=det2, name='detector2')
    diagram.add_neutrons(
        Lmax=1.05 * det2, lambda_min=lambda_min, time_offset=frame_offset
    )
    diagram.add_neutron(
        L=det1,
        time_offset=frame_offset,
        wavelength=20.0 * sc.units.angstrom,
        label=r'$T_0^{i+1}+\Delta t$',
    )

    props = {'arrowstyle': '-|>'}
    x0 = diagram.frame_length
    x1 = diagram.frame_length + frame_offset
    m = sc.Unit('m')
    ms = sc.Unit('ms')
    diagram.annotate(
        r'$T_0^i$', xy=(x0, 0 * m), xytext=(x0 - 20 * ms, 3 * m), arrowprops=props
    )
    diagram.annotate(
        r'$T_0^i+\Delta T_0$',
        xy=(x1, 0 * m),
        xytext=(x1 - 20 * ms, 5 * m),
        arrowprops=props,
    )
    diagram.annotate(
        r'$T_0^{i+1}+t_{\mathrm{pivot}}(\mathrm{det1})$',
        xy=(205 * ms, det1),
        xytext=(150 * ms, 10 * m),
        arrowprops=props,
    )
    diagram.annotate(
        r'$T_0^{i+2}+t_{\mathrm{pivot}}(\mathrm{det2})$',
        xy=(235 * ms, det2),
        xytext=(220 * ms, 15 * m),
        arrowprops=props,
    )

    # Pivot line
    diagram.add_neutron(
        time_offset=x1, L=1.1 * det2, wavelength=lambda_min, color='black', ls='dashed'
    )

    L = diagram.frame_length.value
    ax.axvspan(2 * L, 3 * L, facecolor='grey', alpha=0.2)

    return fig


def frame_skipping_diagram(tmax=None):
    fig, ax = plt.subplots(1, 1)
    diagram = TimeDistanceDiagram(
        ax, tmax=500 * sc.Unit('ms') if tmax is None else tmax
    )
    det1 = 32.0 * sc.Unit('m')
    det2 = 40.0 * sc.Unit('m')
    lambda_min = 16.0 * sc.units.angstrom
    frame_offset = diagram.to_time(1.5 * sc.Unit('ms'))
    diagram.add_source_pulse()
    diagram.add_sample(distance=20.0 * sc.Unit('m'))
    diagram.add_detector(distance=det1, name='detector1')
    diagram.add_detector(distance=det2, name='detector2')
    diagram.add_neutrons(
        Lmax=1.05 * det2, lambda_min=lambda_min, time_offset=frame_offset, stride=2
    )

    props = {'arrowstyle': '-|>'}
    x0 = 2 * diagram.frame_length
    x1 = 2 * diagram.frame_length + frame_offset
    m = sc.Unit('m')
    ms = sc.Unit('ms')
    diagram.annotate(
        r'$T_0^i$', xy=(x0, 0 * m), xytext=(x0 - 20 * ms, 3 * m), arrowprops=props
    )
    diagram.annotate(
        r'$T_0^i+\Delta T_0$', xy=(x1, 0 * m), xytext=(x1, 5 * m), arrowprops=props
    )

    return fig
