"""Make plots of the computed times for disk choppers."""

from dataclasses import replace
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


def plot_anticlockwise(ax, ch):
    theta_tilde = ch.beam_position.value
    omega = ch.angular_frequency.value

    theta = sc.linspace('theta', -pi / 2, 3.2 * pi, 100, unit='rad')
    time = ch.time_offset_angle_at_beam(angle=theta)
    time_wrapped = time % (1 / ch.rotation_speed)

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
    ax.set_xlabel(fr'$\theta$ [{theta.unit}]')
    ax.set_ylabel(fr'$\Delta t_g(\theta)$ [{time.unit}]')
    ax.set_xticks(x_ticks, x_tick_labels)
    ax.set_yticks(y_ticks, y_tick_labels)


def plot_clockwise(ax, ch):
    theta_tilde = ch.beam_position.value
    omega = abs(ch.angular_frequency.value)

    theta = sc.linspace('theta', -pi / 2, 3.2 * pi, 100, unit='rad')
    time = ch.time_offset_angle_at_beam(angle=theta)
    time_wrapped = time % (1 / abs(ch.rotation_speed))

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
    ax.set_xlabel(fr'$\theta$ [{theta.unit}]')
    ax.set_ylabel(fr'$\Delta t_g(\theta)$ [{time.unit}]')
    ax.set_xticks(x_ticks, x_tick_labels)
    ax.set_yticks(y_ticks, y_tick_labels)


def plot(ch, name):
    fig, axs = plt.subplots(1, 2, layout="constrained", figsize=(11, 5))

    axs[0].set_title('Clockwise')
    plot_clockwise(axs[0], replace(ch, rotation_speed=-ch.rotation_speed))

    axs[1].set_title('Anticlockwise')
    assert not ch.is_clockwise  # nosec: B101
    plot_anticlockwise(axs[1], ch)

    fig.savefig(OUT_DIR.joinpath(name).with_suffix('.svg'))


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    ch = DiskChopper(
        position=sc.vector([0.0, 0.0, 0.0], unit='m'),
        rotation_speed=sc.scalar(2.3, unit='Hz'),
        beam_position=sc.scalar(2.5, unit='rad'),
        phase=sc.scalar(0.0, unit='rad'),
        slit_edges=sc.empty(sizes={'slit': 0, 'edge': 2}, unit='rad'),
    )

    set_light_style()
    plot(ch, 'disk-chopper-time-curve')
    set_dark_style()
    plot(ch, 'disk-chopper-time-curve-dark')


if __name__ == "__main__":
    main()
