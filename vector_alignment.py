from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipp as sc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.linalg import polar
from scipy.spatial.transform import Rotation

from scippneutron.spectroscopy.goniometer import Goniometer, GoniometerAxis


def main() -> None:
    axes = [
        GoniometerAxis(
            name="gcu",
            axis='+x',
            min=sc.scalar(-30, unit="degree"),
            max=sc.scalar(30, unit="degree"),
        ),
        GoniometerAxis(
            name="gcl",
            axis='+z',
            min=sc.scalar(-30, unit="degree"),
            max=sc.scalar(30, unit="degree"),
        ),
        GoniometerAxis(
            name=r"$\omega$",
            axis='+y',
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


if __name__ == "__main__":
    main()
