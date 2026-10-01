from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipp as sc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.linalg import polar
from scipy.spatial.transform import Rotation


def main() -> None:
    axes = [
        GoniometerAxis(
            name="gcu",
            axis=sc.vector([1, 0, 0]),
            min=sc.scalar(-30, unit="degree"),
            max=sc.scalar(30, unit="degree"),
        ),
        GoniometerAxis(
            name="gcl",
            axis=sc.vector([0, 0, 1]),
            min=sc.scalar(-30, unit="degree"),
            max=sc.scalar(30, unit="degree"),
        ),
        GoniometerAxis(
            name=r"$\omega$",
            axis=sc.vector([0, 1, 0]),
            min=sc.scalar(-np.inf, unit="degree"),
            max=sc.scalar(np.inf, unit="degree"),
        ),
    ]
    goni = Goniometer(axes)
    goni.plot()
    plt.show()
    return

    kf = 1.2
    ki = 1.4
    theta = 0.4
    Q = np.sqrt(ki**2 + kf**2 - 2 * ki * kf * np.cos(theta))
    r = (
        np.array(
            [
                [-kf * np.sin(theta), ki - kf * np.cos(theta), 0],
                [0, 0, Q],
                [ki - kf * np.cos(theta), kf * np.sin(theta), 0],
            ]
        )
        / Q
    )

    # u, p = polar(r)
    # print("A", a)
    # print("U", u)
    # print("P", p)
    # print(np.linalg.det(p))  # no scaling if det(p) = 1

    # rot = Rotation.from_matrix(a)  # extracts orthogonal part from a
    # print(rot)
    # print(rot.as_quat())
    # print(rot.magnitude())

    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 2])
    U = np.eye(3)
    B = np.eye(3)

    ubv1 = U @ B @ v1
    ubv2 = U @ B @ v2
    t1 = ubv1 / np.linalg.norm(ubv1)
    t3 = np.cross(ubv1, ubv2) / np.linalg.norm(np.cross(ubv1, ubv2))
    t2 = np.cross(t3, t1)
    T = np.c_[t1, t2, t3]

    target = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    # --- determine rotation ---

    savici = target @ np.linalg.inv(T)
    # print(savici)
    # print((savici @ T).round())

    # Use SciPy
    #   This is simple and works with multiple rotations and doesn't need to
    #   build the rotation matrix. But it is also an iterative, approximate procedure.
    #   See the error `rssd`. Don't know if the above procedure works better.
    #   At least, it should not use a matrix inverse...
    # Transpose to match expected layout
    rot, rssd = Rotation.align_vectors(target.T, T.T)
    sp = rot.as_matrix()
    # print(sp)
    # print((sp @ T).round())
    # print("rssd (error)", rssd)

    # --- determine angles ---

    R = savici

    seq = 'YZX'  # capital letters!
    # TODO handle warning for gimbal lock?
    angles = Rotation.from_matrix(R).as_euler(seq)
    angle_x = angles[seq.index('X')]
    angle_y = angles[seq.index('Y')]
    angle_z = angles[seq.index('Z')]
    print(angle_x, angle_y, angle_z)


class GoniometerAxis:
    __slots__ = ("axis", "limits", "name")

    def __init__(
        self, *, name: str, axis: sc.Variable, min: sc.Variable, max: sc.Variable
    ) -> None:
        self.name = name
        self.axis = axis / sc.norm(axis)
        self.limits = (min, sc.to_unit(max, min.unit))

    def __str__(self) -> str:
        return f"GoniometerAxis({self.name!r}, axis={self.axis.value}, limits=({self.limits[0]:c}, {self.limits[1]:c}))"

    def __repr__(self) -> str:
        return str(self)


class Goniometer:
    def __init__(self, axes: Iterable[GoniometerAxis]) -> None:
        self.axes = list(axes)

    def plot(self, *, ax: Axes3D | None = None) -> plt.Figure | None:
        if ax is None:
            fig = plt.figure()
            ax: Axes3D = fig.add_subplot(projection='3d')  # type: ignore[assignment]
        else:
            fig = None

        ax.view_init(elev=None, azim=-135, vertical_axis="y")

        x, y, z = 0, 0, 0
        ax.quiver(x, y, z, 1, 0, 0, length=1, arrow_length_ratio=0.20, colors='C1')
        ax.quiver(x, y, z, 0, 1, 0, length=1, arrow_length_ratio=0.20, colors='C2')
        ax.quiver(x, y, z, 0, 0, 1, length=1, arrow_length_ratio=0.20, colors='C0')

        ax.text(1.1, 0, 0, "x", color='C1')
        ax.text(0, 1.1, 0, "y", color='C2')
        ax.text(0, 0, 1.1, "z", color='C0')

        draw_goniometer_axis(ax, axis=[0, 1, 0], y=-2, color='C0', name=r"$\omega$")
        draw_goniometer_axis(ax, axis=[0, 0, 1], y=-1.3, color='C1', name="gcl")
        draw_goniometer_axis(ax, axis=[1, 0, 0], y=-0.6, color='C2', name="gcu")

        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")

        ax.set_xlim(-2, 2)
        ax.set_zlim(-2, 2)
        ax.set_ylim(-2.3, 1.3)

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_ticks([])
            # axis.line.set_visible(False)

        return fig


def draw_goniometer_axis(
    ax: Axes3D,
    *,
    axis: npt.ArrayLike,
    y: float,
    color: str,
    name: str | None = None,
    axis_length: float = 1,
    circle_radius: float = 0.5,
    n_circle_segments: int = 20,
) -> None:
    center = np.array([0, y, 0])
    axis = np.asarray(axis) / np.linalg.norm(axis)

    tail = center - axis * axis_length / 2
    ax.quiver(
        *tail,
        *axis,
        length=axis_length,
        arrow_length_ratio=0.2,
        colors=color,
        alpha=0.7,
        linewidth=2,
    )
    if name is not None:
        ax.text(
            *tail - axis * axis_length / 4,
            name,
            color=color,
            horizontalalignment="center",
            verticalalignment="center",
        )

    draw_arrow_circle(
        ax,
        axis=axis,
        center=center,
        radius=circle_radius,
        n_segments=n_circle_segments,
        color=color,
    )


def draw_arrow_circle(
    ax: Axes3D,
    *,
    axis: npt.ArrayLike,
    center: npt.ArrayLike,
    radius: float,
    n_segments: int,
    color: str,
    arc_fraction: float = 3 / 4,
) -> None:
    axis = np.asarray(axis, dtype=float)
    w = axis / np.linalg.norm(axis)

    # Pick a reference vector that is not (nearly) parallel to the axis.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, w)) > 0.9:
        ref = np.array([0.0, 0.0, 1.0])

    u = np.cross(ref, w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)  # right-handed: u x v = w

    center = np.asarray(center, dtype=float)
    angles = np.linspace(0, arc_fraction * 2 * np.pi, n_segments + 1)
    points = center + radius * (
        np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
    )

    lines = Line3DCollection([points[:-1]], colors=color)
    vec = points[-1] - points[-2]
    ax.quiver(
        *points[-2],
        *vec,
        length=np.linalg.norm(vec),
        arrow_length_ratio=15,
        colors=color,
    )
    ax.add_artist(lines)


if __name__ == "__main__":
    main()
