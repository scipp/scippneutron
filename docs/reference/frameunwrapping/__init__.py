import scipp as sc
import matplotlib.pyplot as plt
from scippneutron.tof import TimeDistanceDiagram


def default_frame_diagram(tmax=300 * sc.Unit('ms')):
    fig, ax = plt.subplots(1, 1)
    diagram = TimeDistanceDiagram(ax, tmax=tmax)
    det1 = 32.0 * sc.Unit('m')
    det2 = 40.0 * sc.Unit('m')
    frame_offset = diagram.to_time(1.5 * sc.Unit('ms'))
    diagram.add_source_pulse()
    diagram.add_sample(distance=20.0 * sc.Unit('m'))
    diagram.add_detector(distance=det1, name='detector1')
    diagram.add_detector(distance=det2, name='detector2')
    diagram.add_neutrons(Lmax=1.05 * det2,
                         lambda_min=16.0 * sc.units.angstrom,
                         time_offset=frame_offset)
    diagram.add_neutron(L=det1,
                        time_offset=frame_offset,
                        wavelength=20.0 * sc.units.angstrom,
                        label=r'$T_0^{i+1}+\Delta t$')

    props = dict(arrowstyle='-|>')
    x0 = diagram.frame_length
    x1 = diagram.frame_length + frame_offset
    ax.annotate(r'$T_0^i$',
                xy=(x0.value, 0),
                xytext=(x0.value - 20, 3),
                arrowprops=props)
    ax.annotate(r'$T_0^i+\Delta T_0$',
                xy=(x1.value, 0),
                xytext=(x1.value, 5),
                arrowprops=props)
    ax.annotate(r'$T_0^{i+1}+t_{\mathrm{pivot}}(\mathrm{det1})$',
                xy=(200, det1.value),
                xytext=(150, 10),
                arrowprops=props)
    ax.annotate(r'$T_0^{i+2}+t_{\mathrm{pivot}}(\mathrm{det2})$',
                xy=(230, det2.value),
                xytext=(220, 15),
                arrowprops=props)

    # Pivot line
    ax.plot([x1.value, x1.value + 177], [0, 1.1 * det2.value],
            marker='',
            color='black',
            ls='dashed')

    return fig


def frame_skipping_diagram(tmax=500 * sc.Unit('ms')):
    fig, ax = plt.subplots(1, 1)
    diagram = TimeDistanceDiagram(ax, tmax=tmax)
    det1 = 32.0 * sc.Unit('m')
    det2 = 40.0 * sc.Unit('m')
    frame_offset = diagram.to_time(1.5 * sc.Unit('ms'))
    diagram.add_source_pulse()
    diagram.add_sample(distance=20.0 * sc.Unit('m'))
    diagram.add_detector(distance=det1, name='detector1')
    diagram.add_detector(distance=det2, name='detector2')
    diagram.add_neutrons(Lmax=1.05 * det2,
                         lambda_min=16.0 * sc.units.angstrom,
                         time_offset=frame_offset,
                         stride=2)

    props = dict(arrowstyle='-|>')
    x0 = 2 * diagram.frame_length
    x1 = 2 * diagram.frame_length + frame_offset
    ax.annotate(r'$T_0^i$',
                xy=(x0.value, 0),
                xytext=(x0.value - 20, 3),
                arrowprops=props)
    ax.annotate(r'$T_0^i+\Delta T_0$',
                xy=(x1.value, 0),
                xytext=(x1.value, 5),
                arrowprops=props)

    return fig
