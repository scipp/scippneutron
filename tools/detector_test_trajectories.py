"""Plots of detector trajectories used in single crystal normalization tests."""

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    n = len(CASES)
    _, axs = plt.subplots(
        int(np.ceil(n / 3)), 3, figsize=(15, 10), layout='constrained'
    )
    for ax, case in zip(axs.flatten(), CASES, strict=False):
        case(ax)
    plt.show()


def case_a(ax: plt.Axes) -> None:
    ax.set_title("A1 / A2")
    h_min1, h_max1 = 0.1, 0.9
    mom_min1, mom_max1 = 1.0, 1.5
    h_min2, h_max2 = 0.5, 0.8
    mom_min2, mom_max2 = 0.9, 1.4

    h_edges = np.array([-0.1, 0.3, 0.7, 1.0, 1.3])
    mom_edges = np.array([0.5, 0.9, 1.3, 1.6])

    draw_2d_projection(
        ax,
        [[h_min1, mom_min1], [h_min2, mom_min2]],
        [[h_max1, mom_max1], [h_max2, mom_max2]],
        h_edges,
        mom_edges,
        "$h$",
        "$k_f$",
    )

    a1 = 0.125
    b1 = 0.175
    c1 = 0.075
    d1 = 0.125

    a2 = 1 / 3
    b2 = 2 / 30
    c2 = 0.1

    yline(ax, 0.3, mom_min1, mom_min1 + a1, "$a_1$")
    yline(ax, 0.3, 1.3 - b1, 1.3, "$b_1$")
    yline(ax, 0.3, 1.3, 1.3 + c1, "$c_1$")
    yline(ax, 0.3, mom_max1 - d1, mom_max1, "$d_1$")

    yline(ax, 1.0, mom_min2, mom_min2 + a2, "$a_2$")
    yline(ax, 1.0, 1.3 - b2, 1.3, "$b_2$")
    yline(ax, 1.0, mom_max2 - c2, mom_max2, "$c_2$")

    xline(ax, mom_min1, h_min1, 0.3, None, ls='--', c='gray')
    xline(ax, mom_max1, 0.3, h_max1, None, ls='--', c='gray')
    xline(ax, 1.3 + c1, 0.3, 0.7, None, ls='--', c='gray')

    xline(ax, mom_min2, h_min2, 1.0, None, ls='--', c='gray')
    xline(ax, mom_min2 + a2, 0.7, 1.0, None, ls='--', c='gray')
    xline(ax, mom_max2, h_max2, 1.0, None, ls='--', c='gray')


def case_b(ax: plt.Axes) -> None:
    ax.set_title("B")
    h_min, h_max = 1.22, 1.35
    mom_min, mom_max = 1.1, 0.1

    h_edges = np.array([0.9, 1.0, 1.2, 1.3])
    mom_edges = np.array([0.0, 0.2, 0.7, 1.0])

    draw_2d_projection(
        ax,
        [[h_min, mom_min]],
        [[h_max, mom_max]],
        h_edges,
        mom_edges,
        "$h$",
        "$k_f$",
    )

    a = 0.3
    b = 0.21538461538461542

    yline(ax, 1.3, 0.7, 0.7 + a, "$a$")
    yline(ax, 1.3, 0.7 - b, 0.7, "$b$")


def case_c(ax: plt.Axes) -> None:
    ax.set_title("C")
    h_min, h_max = 1.0, 0.6
    mom_min, mom_max = 0.9, 0.3

    h_edges = np.array([-0.1, 0.3, 0.7, 1.1, 1.5, 1.9])
    mom_edges = np.array([0.4, 0.5, 1.0, 1.1])

    draw_2d_projection(
        ax,
        [[h_min, mom_min]],
        [[h_max, mom_max]],
        h_edges,
        mom_edges,
        "$h$",
        "$k_f$",
    )

    a = 0.4
    b = 0.05
    c = 0.05

    yline(ax, 0.7, mom_min, mom_min - a, "$a$")
    yline(ax, 0.7, 0.5, 0.5 - b, "$b$")
    yline(ax, 0.7, 0.4, 0.4 + c, "$c$")

    xline(ax, mom_min, h_min, 0.7, None, ls='--', c='gray')


def case_d(ax: plt.Axes) -> None:
    ax.set_title("D")
    h_min, h_max = 0.6, 0.4
    mom_min, mom_max = 1.0, 1.2

    h_edges = np.array([-0.1, 0.3, 0.7, 1.0, 1.3])
    mom_edges = np.array([0.5, 0.9, 1.3, 1.6])

    draw_2d_projection(
        ax,
        [[h_min, mom_min]],
        [[h_max, mom_max]],
        h_edges,
        mom_edges,
        "$h$",
        "$k_f$",
    )

    a = 0.2

    yline(ax, h_min, mom_min, mom_min + a, "$a$")
    xline(ax, mom_max, h_max, h_min, None, ls='--', c='gray')


def case_e(ax: plt.Axes) -> None:
    ax.set_title("E")
    h_min, h_max = 0.5, 0.5
    mom_min, mom_max = 0.6, 1.4

    h_edges = np.array([-0.1, 0.3, 0.7, 1.0, 1.3])
    mom_edges = np.array([0.5, 0.9, 1.3, 1.6])

    draw_2d_projection(
        ax,
        [[h_min, mom_min]],
        [[h_max, mom_max]],
        h_edges,
        mom_edges,
        "$h$",
        "$k_f$",
    )

    a = 0.3
    b = 0.4
    c = 0.1

    yline(ax, h_min, mom_min, mom_min + a, "$a$")
    yline(ax, h_min, 0.9, 0.9 + b, "$b$")
    yline(ax, h_min, mom_max - c, mom_max, "$c$")


def case_f(ax: plt.Axes) -> None:
    ax.set_title("F")
    h_min, h_max = 0.3, 0.8
    mom_min, mom_max = 0.7, 1.3

    h_edges = np.array([-0.1, 0.3, 0.7, 1.0, 1.3])
    mom_edges = np.array([0.5, 0.9, 1.3, 1.6])

    draw_2d_projection(
        ax,
        [[h_min, mom_min]],
        [[h_max, mom_max]],
        h_edges,
        mom_edges,
        "$h$",
        "$k_f$",
    )

    a = 0.2
    b = 0.28
    c = 0.12

    yline(ax, 0.7, mom_min, mom_min + a, "$a$")
    yline(ax, 0.7, 0.9, 0.9 + b, "$b$")
    yline(ax, 0.7, 1.3 - c, 1.3, "$c$")
    xline(ax, mom_min, h_min, 0.7, None, ls='--', c='gray')
    xline(ax, mom_max, h_max, 0.7, None, ls='--', c='gray')


CASES = (case_a, case_b, case_c, case_d, case_e, case_f)


def xline(ax, y, xmin, xmax, label, **kwargs):
    kwargs.setdefault("c", 'k')
    ax.plot([xmin, xmax], [y, y], **kwargs)
    if label:
        ax.text((xmin + xmax) / 2, y, label, ha='center', va='bottom')


def yline(ax, x, ymin, ymax, label):
    ax.plot([x, x], [ymin, ymax], c='k')
    ax.text(x, (ymax + ymin) / 2, label, ha='left', va='center')


def draw_2d_projection(ax, start, stop, x_edges, y_edges, xlabel, ylabel):
    for edge in x_edges:
        ax.plot([edge, edge], [y_edges[0], y_edges[-1]], c='0.8')
    for edge in y_edges:
        ax.plot([x_edges[0], x_edges[-1]], [edge, edge], c='0.8')
    ax.set_xticks(x_edges)
    ax.set_yticks(y_edges)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for a, b in zip(start, stop, strict=True):
        ax.plot([a[0], b[0]], [a[1], b[1]], marker='o')


if __name__ == "__main__":
    main()
