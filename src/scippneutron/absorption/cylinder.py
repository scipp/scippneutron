from dataclasses import dataclass
from typing import Literal

import numpy as np
import quadratures
import scipp as sc
from numpy.polynomial.legendre import leggauss


@dataclass
class Cylinder:
    symmetry_line: sc.Variable
    center_of_base: sc.Variable
    radius: sc.Variable
    height: sc.Variable

    def beam_intersection(self, start_point, direction):
        'Length of intersection between beam and cylinder'
        base_point = self.center_of_base - start_point
        cyl_intersection, cyl_left, cyl_right = _line_infinite_cylinder_intersection(
            self.symmetry_line, base_point, self.radius, direction
        )
        sla_intersection, sla_left, sla_right = _line_slab_intersection(
            self.symmetry_line, base_point, self.height, direction
        )
        return sc.where(
            cyl_intersection & sla_intersection,
            _positive_interval_intersection(
                (sla_left, sla_right), (cyl_left, cyl_right)
            ),
            sc.scalar(0.0, unit=start_point.unit),
        )

    @property
    def center(self):
        return self.center_of_base + self.symmetry_line * self.height / 2

    def _select_quadrature_points(self, kind):
        if kind == 'expensive':
            x, w = leggauss(max(round(10 * self.height / self.radius), 10))
            return _cylinder_quadrature_from_product(
                quadratures.circle254_cheb,
                dict(x=x, weights=w),  # noqa: C408
            )
        if kind == 'medium':
            x, w = leggauss(max(round(10 * self.height / self.radius), 5))
            # Would be nice to have a medium size Chebychev quadrature on the disk,
            # but I only found the large one for now.
            return _cylinder_quadrature_from_product(
                quadratures.circle54,
                dict(x=x, weights=w),  # noqa: C408
            )
        if kind == 'cheap':
            x, w = leggauss(max(round(5 * self.height / self.radius), 5))
            return _cylinder_quadrature_from_product(
                quadratures.circle12,
                dict(x=x, weights=w),  # noqa: C408
            )
        raise NotImplementedError

    def quadrature(self, kind: Literal['expensive', 'medium', 'cheap']):
        quad = self._select_quadrature_points(kind)
        # Scale to size of cylinder
        quad['x'] = quad['x'] * self.radius
        quad['y'] = quad['y'] * self.radius
        quad['z'] = quad['z'] * self.height / 2
        quad['weights'] = quad['weights'] * (self.radius**2 * self.height / 2)
        quad_points = sc.vectors(
            dims=['quad'],
            values=(
                sc.concat(
                    [sc.array(dims=['quad'], values=quad[x]) for x in 'xyz'], dim='row'
                )
                .transpose(['quad', 'row'])
                .values
            ),
        )
        # By default the cylinder quadrature has z as the symmetry axis.
        # We need to rotate the quadrature so the symmetry axis matches the cylinder.
        u = sc.cross(sc.vector([0, 0, 1]), self.symmetry_line)
        un = sc.norm(u)
        if un >= 1e-10:
            u *= sc.asin(un) / un
            quad_points = sc.spatial.rotations_from_rotvecs(u) * quad_points
        # By default the cylinder quadrature center is at the origin.
        # We need to move it so the center matches the cylinder.
        quad_points += self.center
        return (quad_points, sc.array(dims=['quad'], values=quad['weights']))


def _cylinder_quadrature_from_product(disk_quadrature, line_quadrature):
    return dict(  # noqa: C408
        weights=np.array(
            [
                disk_w * line_w
                for disk_w in disk_quadrature['weights']
                for line_w in line_quadrature['weights']
            ]
        ),
        x=np.repeat(disk_quadrature['x'], len(line_quadrature['weights'])),
        y=np.repeat(disk_quadrature['y'], len(line_quadrature['weights'])),
        z=np.tile(line_quadrature['x'], len(disk_quadrature['weights'])),
    )


def _max0(x):
    return sc.where(x >= sc.scalar(0.0, unit=x.unit), x, sc.scalar(0.0, unit=x.unit))


def _minimum(x, y):
    return sc.where(x <= y, x, y)


def _maximum(x, y):
    return sc.where(x >= y, x, y)


def _positive_intersection_sorted_by_left(a, b):
    '''Length of the intersection of a and b and the positive real axis,
    provided that the leftmost point of a is left of the leftmost point of b'''
    return sc.where(
        b[0] <= a[1],
        _max0(_minimum(b[1], a[1])) - _max0(b[0]),
        sc.scalar(0.0, unit=a[0].unit),
    )


def _positive_interval_intersection(a, b):
    '''Length of the intersection of a and b and the positive real axis'''
    return sc.where(
        a[0] <= b[0],
        _positive_intersection_sorted_by_left(a, b),
        _positive_intersection_sorted_by_left(b, a),
    )


def _line_infinite_cylinder_intersection(a, b, r, n):
    '''Intersection between a cylinder and a line through the origin with direction n.
    Returns:
        intersection: bool, True if the line and the cylinder intersect anywhere
        left: float,
            distance from origin to left edge
            of intersection segment (direction opposite n)
        right: float,
            distance from origin to right edge
            of intersection segment (direction n)
    '''
    nxa = sc.cross(n, a)
    nxa_square = sc.dot(nxa, nxa)
    s2 = nxa_square * r**2 - sc.dot(b, nxa) ** 2
    s = sc.sqrt(s2)
    m = sc.dot(nxa, sc.cross(b, a))
    intersection = s2 >= sc.scalar(0.0, unit=s2.unit)
    left = (m - s) / nxa_square
    right = (m + s) / nxa_square
    origin_in_cylinder = sc.norm(b - sc.dot(b, a) * a) <= r
    ndota = sc.dot(n, a)
    parallel_to_cylinder = sc.abs(ndota) == sc.scalar(1.0, unit=ndota.unit)
    return (
        sc.where(parallel_to_cylinder, origin_in_cylinder, intersection),
        sc.where(parallel_to_cylinder, sc.scalar(float('-inf'), unit=left.unit), left),
        sc.where(parallel_to_cylinder, sc.scalar(float('inf'), unit=right.unit), right),
    )


def _line_slab_intersection(a, b, h, n):
    '''Intersection between a slab (the volume between two parallel planes)
    and a line through the origin with direction n.
    Returns:
        intersection: bool, True if the line and the slab intersect anywhere
        left: float,
            distance from origin to left edge
            of intersection segment (direction opposite n)
        right: float,
            distance from origin to right edge
            of intersection segment (direction n)
    '''

    ndota = sc.dot(n, a)
    bdota = sc.dot(b, a)
    origin_in_plane = (bdota <= 0) & (bdota >= -h)
    parallel_to_slab = sc.abs(ndota) == sc.scalar(0, unit=ndota.unit)
    t0 = bdota / ndota
    t1 = t0 + h / ndota
    left = _minimum(t0, t1)
    right = _maximum(t1, t0)
    return (
        origin_in_plane | (~parallel_to_slab),
        sc.where(
            parallel_to_slab,
            sc.scalar(float('-inf'), unit=b.unit),
            left,
        ),
        sc.where(
            parallel_to_slab,
            sc.scalar(float('inf'), unit=b.unit),
            right,
        ),
    )
