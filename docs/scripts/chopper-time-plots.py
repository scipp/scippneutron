"""Make plots of the computed times for disk choppers."""

from dataclasses import replace
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import scipp as sc
from matplotlib.patches import Rectangle
from numpy import pi

from scippneutron.chopper import DiskChopper

OUT_DIR = Path(__file__).resolve().parent.parent / '_static' / 'chopper'

GENERAL_STYLE = {
    'axes.labelsize': 'xx-large',
    'axes.titlesize': 'xx-large',
    'xtick.major.size': 5,
    'xtick.labelsize': 'x-large',
    'ytick.major.size': 5,
    'ytick.labelsize': 'x-large',
}


def set_light_style():
    plt.style.use(
        [
            'default',
            GENERAL_STYLE,
            {
                # High jack the color cycle to get theme-specific grid colors.
                'axes.prop_cycle': plt.cycler(
                    'color', ['#1f77b4', '#ff7f0e', '0.2', '0.5']
                ),
            },
        ]
    )


def set_dark_style():
    plt.style.use(
        [
            'dark_background',
            GENERAL_STYLE,
            {
                # Colors from the pydata-sphinx-theme.
                'axes.edgecolor': 'c3d6dd',
                'axes.facecolor': '#14181e',
                'axes.labelcolor': 'c3d6dd',
                'figure.edgecolor': '#14181e',
                'figure.facecolor': '#14181e',
                'savefig.edgecolor': '#14181e',
                'savefig.facecolor': '#14181e',
                'text.color': 'c3d6dd',
                # High jack the color cycle to get theme-specific grid colors.
                'axes.prop_cycle': plt.cycler(
                    'color', ['#8dd3c7', '#feffb3', '0.8', '0.5']
                ),
            },
        ]
    )


def plot_axes_lines(ax, x_ticks, y_ticks):
    for x in x_ticks:
        ax.axvline(x=x, color='C2' if x == 0 else 'C3', ls=':')
    for y in y_ticks:
        ax.axhline(y=y, color='C2' if y == 0 else 'C3', ls=':')


def plot_t_vs_angle_anticlockwise(ax, ch):
    theta_tilde = ch.beam_angle.value
    omega = ch.angular_frequency.value

    theta = sc.linspace('theta', -pi / 2, 3.2 * pi, 100, unit='rad')
    time = ch.time_offset_angle_at_beam(angle=theta)
    time_wrapped = time % (1 / ch.frequency)

    x_ticks = [0.0, theta_tilde, 2 * pi, 2 * pi + theta_tilde]
    x_tick_labels = ['0', r'$\tilde{\theta}$', r'$2\pi$', r'$2\pi+\tilde{\theta}$']

    y_ticks = [0.0, theta_tilde / omega, 2 * pi / omega, (2 * pi + theta_tilde) / omega]
    y_tick_labels = [
        '0',
        r'$\frac{\tilde{\theta}}{\omega}$',
        r'$\frac{2\pi}{\omega}$',
        r'$\frac{2\pi+\tilde{\theta}}{\omega}$',
    ]

    ax.add_patch(
        Rectangle(
            (0, theta_tilde / omega), 2 * pi, 2 * pi / omega, facecolor='#80808040'
        )
    )
    plot_axes_lines(ax, x_ticks, y_ticks)
    ax.plot(theta.values[[0, -1]], time.values[[0, -1]], linewidth=2)
    ax.plot(theta.values, time_wrapped.values, linewidth=2)
    ax.set_xlabel(rf'$\theta$ [{theta.unit}]')
    ax.set_ylabel(rf'$\Delta t_g(\theta)$ [{time.unit}]')
    ax.set_xticks(x_ticks, x_tick_labels)
    ax.set_yticks(y_ticks, y_tick_labels)


def plot_t_vs_angle_clockwise(ax, ch):
    theta_tilde = ch.beam_angle.value
    omega = abs(ch.angular_frequency.value)

    theta = sc.linspace('theta', -pi / 2, 3.2 * pi, 100, unit='rad')
    time = ch.time_offset_angle_at_beam(angle=theta)
    time_wrapped = time % (1 / abs(ch.frequency))

    x_ticks = [0, theta_tilde, 2 * pi, 2 * pi + theta_tilde]
    x_tick_labels = ['0', r'$\tilde{\theta}$', r'$2\pi$', r'$2\pi + \tilde{\theta}$']

    y_ticks = [
        -theta_tilde / omega,
        0.0,
        (2 * pi - theta_tilde) / omega,
        2 * pi / omega,
    ]
    y_tick_labels = [
        r'$-\frac{\tilde{\theta}}{|\omega|}$',
        '0',
        r'$\frac{2\pi - \tilde{\theta}}{|\omega|}$',
        r'$\frac{2\pi}{|\omega|}$',
    ]

    ax.add_patch(
        Rectangle(
            (0, -theta_tilde / omega), 2 * pi, 2 * pi / omega, facecolor='#80808040'
        )
    )
    plot_axes_lines(ax, x_ticks, y_ticks)
    ax.plot(theta.values[[0, -1]], time.values[[0, -1]], linewidth=2)
    ax.plot(theta.values, time_wrapped.values, linewidth=2)
    ax.set_xlabel(r'$\theta$ [rad]')
    ax.set_ylabel(r'$\Delta t_g(\theta)$')
    ax.set_xticks(x_ticks, x_tick_labels)
    ax.set_yticks(y_ticks, y_tick_labels)


def plot_t_vs_angle(ch, name):
    fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(11, 5))

    axs[0].set_title('Clockwise')
    plot_t_vs_angle_clockwise(axs[0], replace(ch, frequency=-ch.frequency))

    axs[1].set_title('Anticlockwise')
    assert not ch.is_clockwise  # noqa: S101
    plot_t_vs_angle_anticlockwise(axs[1], ch)

    fig.savefig(OUT_DIR.joinpath(name).with_suffix('.svg'))


def plot_openings_for_multiple(ax, y, ch, pulse_frequency, n, clockwise):
    ch = replace(ch, frequency=n * pulse_frequency * (-1 if clockwise else 1))
    open_times = ch.time_offset_open(pulse_frequency=pulse_frequency).to(unit='s')
    close_times = ch.time_offset_close(pulse_frequency=pulse_frequency).to(unit='s')
    for i, o, c in zip(cycle((0, 1)), open_times, close_times):
        ax.plot([o.value, c.value], [y, y], lw=10, c=f'C{i}', solid_capstyle='butt')


def plot_pulses(ax, period, clockwise):
    if clockwise:
        x = [-period, 0, period, 2 * period]
        ax.plot([-period, 0], [0, 0], lw=10, c='C3', solid_capstyle='butt')
        ax.plot([0, period], [0, 0], lw=10, c='C2', solid_capstyle='butt')
        ax.plot([period, period * 2], [0, 0], lw=10, c='C3', solid_capstyle='butt')
        ax.set_xticks(
            x, [r'$T_0 - \Delta T$', '$T_0$', r'$T_0 + \Delta T$', r'$T_0 + 2\Delta T$']
        )
    else:
        x = [0, period, 2 * period, 3 * period]
        ax.plot([0, period], [0, 0], lw=10, c='C2', solid_capstyle='butt')
        ax.plot(
            [period, period * 2, period * 3],
            [0, 0, 0],
            lw=10,
            c='C3',
            solid_capstyle='butt',
        )
        ax.set_xticks(
            x,
            ['$T_0$', r'$T_0 + \Delta T$', r'$T_0 + 2\Delta T$', r'$T_0 + 3\Delta T$'],
        )
    ax.plot(x, [0] * len(x), ls='', markersize=20, marker='|', c='C2')


def plot_openings_clockwise(ax, ch, pulse_frequency, clockwise):
    pulse_frequency = pulse_frequency.to(unit='Hz')
    period = sc.reciprocal(pulse_frequency).value
    plot_pulses(ax, period, clockwise)

    ns = [0.5, 1, 2]
    ys = []
    for i, n in enumerate(ns, 1):
        ys.append(0.1 * i)
        plot_openings_for_multiple(ax, ys[-1], ch, pulse_frequency, n, clockwise)
    ax.set_yticks(ys, list(map(str, ns)))

    ax.set_xlabel('$t_g$')
    ax.set_ylabel('frequency ratio')
    ax.set_ylim((-0.04, 0.34))


def plot_openings(ch, name, pulse_frequency):
    ch = replace(
        ch, beam_angle=sc.scalar(5, unit='deg'), phase=sc.scalar(360 + 50, unit='deg')
    )

    fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(11, 2.5))

    axs[0].set_title('Clockwise')
    plot_openings_clockwise(axs[0], ch, pulse_frequency, True)

    axs[1].set_title('Anticlockwise')
    plot_openings_clockwise(axs[1], ch, pulse_frequency, False)

    fig.savefig(OUT_DIR.joinpath(name).with_suffix('.svg'))


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    ch = DiskChopper(
        axle_position=sc.vector([0.0, 0.0, 0.0], unit='m'),
        frequency=sc.scalar(2.3, unit='Hz'),
        beam_angle=sc.scalar(2.5, unit='rad'),
        phase=sc.scalar(0.0, unit='rad'),
        slit_begin=sc.array(dims=['slit'], values=[10.0, 150.0], unit='deg'),
        slit_end=sc.array(dims=['slit'], values=[70.0, 280.0], unit='deg'),
    )
    pulse_frequency = sc.scalar(10.0, unit='Hz')

    set_light_style()
    # plot_t_vs_angle(ch, 'disk-chopper-time-curve')
    # plot_openings(ch, 'disk-chopper-openings', pulse_frequency)
    set_dark_style()
    # plot_t_vs_angle(ch, 'disk-chopper-time-curve-dark')
    plot_openings(ch, 'disk-chopper-openings-dark', pulse_frequency)


if __name__ == "__main__":
    main()
