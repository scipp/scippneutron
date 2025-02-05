from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipp as sc
from numpy.polynomial.chebyshev import chebgauss
from numpy.polynomial.legendre import leggauss

from . import quadratures
from .types import SampleShape


@dataclass
class Cylinder(SampleShape):
    symmetry_line: sc.Variable
    center_of_base: sc.Variable
    radius: sc.Variable
    height: sc.Variable

    def beam_intersection(
        self, start_point: sc.Variable, direction: sc.Variable
    ) -> sc.Variable:
        """Length of intersection between beam and cylinder"""
        base_point = self.center_of_base - start_point
        cyl_intersection, *cyl_interval = _line_infinite_cylinder_intersection(
            self.symmetry_line, base_point, self.radius, direction
        )
        slab_intersection, *slab_interval = _line_slab_intersection(
            self.symmetry_line, base_point, self.height, direction
        )
        return sc.where(
            cyl_intersection & slab_intersection,
            _positive_interval_intersection(slab_interval, cyl_interval),
            sc.scalar(0.0, unit=start_point.unit),
        )

    @property
    def center(self) -> sc.Variable:
        return self.center_of_base + self.symmetry_line * self.height / 2

    @property
    def volume(self) -> sc.Variable:
        return self.radius**2 * self.height * np.pi

    def _select_quadrature_points(self, kind):
        if kind == 'expensive':
            k = round(max(min(11 * (self.height / self.radius).value, 35), 11))
            x, w = chebgauss(k)
            w *= (1 - x**2) ** 0.5
            w /= sum(w) / 2
            quad = _cylinder_quadrature_from_product(
                quadratures.disk256_cheb,
                dict(x=x, weights=w),  # noqa: C408
            )
        elif kind == 'medium':
            k = round(max(min(7 * (self.height / self.radius).value, 25), 7))
            x, w = chebgauss(k)
            w *= (1 - x**2) ** 0.5
            w /= sum(w) / 2
            # Would be nice to have a medium size Chebychev quadrature on the disk,
            # but I only found the large one for now.
            quad = _cylinder_quadrature_from_product(
                quadratures.disk55,
                dict(x=x, weights=w),  # noqa: C408
            )
        elif kind == 'cheap':
            k = round(max(min(5 * (self.height / self.radius).value, 15), 5))
            x, w = leggauss(k)
            quad = _cylinder_quadrature_from_product(
                quadratures.disk12,
                dict(x=x, weights=w),  # noqa: C408
            )
        elif kind == 'mc' or (
            isinstance(kind, tuple) and len(kind) > 0 and kind[0] == 'mc'
        ):
            # The Monte-Carlo integration option is useful because
            # while inefficient it is guaranteed unbiased and good for debugging.
            k = 5000 if kind == 'mc' or len(kind) == 1 else kind[1]
            # Uniform sampling in cylinder
            r = np.random.random(k) ** 0.5
            th = 2 * np.pi * np.random.random(k)
            z = 2 * np.random.random(k) - 1
            quad = {
                'x': r * np.cos(th),
                'y': r * np.sin(th),
                'z': z,
                'weights': 2 * np.pi * np.ones(k) / k,
            }
        else:
            raise NotImplementedError
        return {k: sc.array(dims=['quad'], values=v) for k, v in quad.items()}

    def quadrature(
        self,
        kind: Literal['expensive', 'medium', 'cheap', 'mc'] | tuple[Literal['mc'], int],
    ) -> tuple[sc.Variable, sc.Variable]:
        'Returns quadrature points and weights of the cylinder.'
        quad = self._select_quadrature_points(kind)
        # Scale to size of cylinder
        x = (quad['x'] * self.radius).to(unit=self.center.unit)
        y = (quad['y'] * self.radius).to(unit=self.center.unit)
        z = (quad['z'] * self.height / 2).to(unit=self.center.unit)
        # Quadrature had total weight 2pi before, now it is 2pi radius^2 * height
        weights = quad['weights'] * (self.radius**2 * self.height / 2)
        points = sc.vectors(
            dims=['quad'],
            values=(sc.concat([x, y, z], dim='row').transpose(['quad', 'row']).values),
            unit=self.center.unit,
        )
        # By default the cylinder quadrature has z as the symmetry axis.
        # We need to rotate the quadrature to match the symmetry axis of the cylinder.
        u = sc.cross(sc.vector([0, 0, 1]), self.symmetry_line)
        un = sc.norm(u)
        if un >= 1e-10:
            u *= sc.asin(un) / un
            points = sc.spatial.rotations_from_rotvecs(u) * points

        # By default the cylinder quadrature center is at the origin.
        # We need to move it so the center matches the cylinder.
        points += self.center
        return points, weights


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


def _minimum(x, y):
    return sc.where(x <= y, x, y)


def _maximum(x, y):
    return sc.where(x >= y, x, y)


def _max0(x):
    return _maximum(x, sc.scalar(0.0, unit=x.unit))


def _positive_interval_intersection(a, b):
    '''Length of the intersection of a and b and the positive real axis'''
    left, right = _maximum(a[0], b[0]), _minimum(a[1], b[1])
    return _max0(_max0(right) - _max0(left))


def _line_infinite_cylinder_intersection(a, b, r, n):
    '''Intersection between a cylinder and a line through the origin with direction n.
    See https://en.wikipedia.org/wiki/Line-cylinder_intersection.

    Parameters:
    -----------
    a: cylinder symmetry line
    b: point on cylinder symmetry line
    r: radius
    n: direction of line to intersect cylinder

    If the cylinder intersects the line through the origin in the direction of n
    then `intersection` is `True` and the cylinder and the line intersects all
    points `t * n` for `left <= t <= right`.

    Returns:
    ---------
    intersection:
        True if the line and the cylinder intersect anywhere
    left:
        first edge of intersection segment (direction opposite n)
    right:
        second edge of intersection segment (direction n)
    '''
    nxa = sc.cross(n, a)
    nxa_square = sc.dot(nxa, nxa)
    parallel_to_cylinder = nxa_square == sc.scalar(0.0, unit=nxa.unit)
    s2 = nxa_square * r**2 - sc.dot(b, nxa) ** 2
    s = sc.sqrt(s2)
    m = sc.dot(nxa, sc.cross(b, a))
    intersection = s2 >= sc.scalar(0.0, unit=s2.unit)
    left = sc.where(
        parallel_to_cylinder,
        sc.scalar(float('-inf'), unit=m.unit),
        (m - s) / nxa_square,
    )
    right = sc.where(
        parallel_to_cylinder, sc.scalar(float('inf'), unit=m.unit), (m + s) / nxa_square
    )
    origin_in_cylinder = sc.norm(b - sc.dot(b, a) * a) <= r
    return (
        sc.where(parallel_to_cylinder, origin_in_cylinder, intersection),
        left,
        right,
    )


def _line_slab_intersection(a, b, h, n):
    '''Intersection between a slab (the volume between two parallel planes)
    and a line through the origin with direction n.

    Returns:
    --------
    intersection:
        True if the line and the slab intersect anywhere
    left:
        first edge of intersection segment (direction opposite n)
    right:
        second edge of intersection segment (direction n)
    '''
    ndota = sc.dot(n, a)
    bdota = sc.dot(b, a)
    origin_in_plane = (bdota <= sc.scalar(0.0, unit=h.unit)) & (bdota >= -h)
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
