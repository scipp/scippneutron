import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_allclose

from scippneutron.absorption.cylinder import Cylinder


@pytest.fixture(
    params=[
        sc.vector([1, 0, 0]),
        sc.vector([0, 1, 0]),
        sc.vector([2**-0.5, 2**-0.5, 0]),
        sc.vector([2**-0.5, -(2**-0.5), 0]),
        sc.vector([-(2**-0.5), 2**-0.5, 0]),
        sc.vector([-(2**-0.5), -(2**-0.5), 0]),
    ]
)
def point_on_unit_circle(request):
    return request.param


@pytest.mark.parametrize('h', [sc.scalar(0.2), sc.scalar(1.0)])
@pytest.mark.parametrize('r', [sc.scalar(0.2), sc.scalar(1.0)])
def test_intersection_in_base(r, h, point_on_unit_circle):
    c = Cylinder(sc.vector([0, 0, 1.0]), sc.vector([0, 0, 0]), r, h)
    assert_allclose(
        c.beam_intersection(point_on_unit_circle, -point_on_unit_circle), 2 * r
    )


@pytest.mark.parametrize('h', [sc.scalar(0.2), sc.scalar(1.0)])
@pytest.mark.parametrize('r', [sc.scalar(0.2), sc.scalar(1.0)])
def test_intersection_diagonal(r, h, point_on_unit_circle):
    ax, base = sc.vector([0, 0, 1.0]), sc.vector([0, 0, 0])
    c = Cylinder(ax, base, r, h)
    x = r * point_on_unit_circle
    n = base - x + ax * h / 2
    assert_allclose(
        c.beam_intersection(x, n / sc.norm(n)), ((2 * r) ** 2 + h**2) ** 0.5
    )


@pytest.mark.parametrize('h', [sc.scalar(0.2), sc.scalar(1.0)])
@pytest.mark.parametrize('r', [sc.scalar(0.2), sc.scalar(1.0)])
def test_intersection_along_axis(r, h, point_on_unit_circle):
    ax, base = sc.vector([0, 0, 1.0]), sc.vector([0, 0, 0])
    c = Cylinder(ax, base, r, h)
    x = (1.0 - np.finfo(float).eps) * r * point_on_unit_circle
    assert_allclose(c.beam_intersection(x, ax), h)


def test_no_intersection(point_on_unit_circle):
    c = Cylinder(
        sc.vector([0, 0, 1.0]), sc.vector([0, 0, 0]), sc.scalar(1.0), sc.scalar(1.0)
    )
    assert_allclose(
        c.beam_intersection(point_on_unit_circle, point_on_unit_circle), sc.scalar(0.0)
    )

    n = point_on_unit_circle - sc.vector([0, 0, 1]) / 2
    n /= sc.norm(n)
    assert_allclose(c.beam_intersection(point_on_unit_circle, n), sc.scalar(0.0))

    x = (1.0 + np.finfo(float).eps) * point_on_unit_circle
    assert_allclose(c.beam_intersection(x, sc.vector([0, 0, 1])), sc.scalar(0.0))


@pytest.mark.parametrize('h', [sc.scalar(0.2), sc.scalar(1.0)])
@pytest.mark.parametrize('r', [sc.scalar(0.2), sc.scalar(1.0)])
def test_intersection_interior(r, h, point_on_unit_circle):
    c = Cylinder(sc.vector([0, 0, 1.0]), sc.vector([0, 0, -h.value / 2]), r, h)
    assert_allclose(c.beam_intersection(sc.vector([0, 0, 0]), point_on_unit_circle), r)


@pytest.mark.parametrize('kind', ['expensive', 'medium', 'cheap'])
@pytest.mark.parametrize(
    ('f', 'expected'),
    [
        ((lambda x: 1.0), sc.scalar(np.pi)),
        ((lambda x: sc.dot(x, sc.vector([0, 0, 1]))), sc.scalar(np.pi / 2)),
        ((lambda x: sc.dot(x, sc.vector([0, 1, 0]))), sc.scalar(0.0)),
        ((lambda x: sc.dot(x, sc.vector([1, 0, 0]))), sc.scalar(0.0)),
        ((lambda x: x.fields.x * x.fields.y), sc.scalar(0.0)),
        (
            (
                lambda x: sc.sin(
                    sc.dot(x, sc.vector([1, 0.5, 0]) * sc.scalar(1, unit='rad'))
                )
                ** 2
            ),
            sc.scalar(0.7972445081889596),
        ),
    ],
)
def test_quadrature(kind, f, expected):
    c = Cylinder(
        sc.vector([0, 0, 1.0]), sc.vector([0, 0, 0.0]), sc.scalar(1.0), sc.scalar(1.0)
    )
    q, w = c.quadrature(kind)
    v = (f(q) * w).sum()
    assert_allclose(v, expected, atol=sc.scalar(1e-8), rtol=sc.scalar(1e-4))


@pytest.mark.parametrize('kind', ['expensive', 'medium', 'cheap'])
@pytest.mark.parametrize(
    'axis', [sc.vector([0, 1, 0.0]), sc.vector([2**-0.5, 0, 2**-0.5])]
)
@pytest.mark.parametrize('base', [sc.vector([0, 0, 0.0]), sc.vector([2.0, 5.3, -4.0])])
@pytest.mark.parametrize('r', [sc.scalar(0.2), sc.scalar(1.0)])
@pytest.mark.parametrize('h', [sc.scalar(0.2), sc.scalar(1.0)])
def test_quadrature_translated_cylinder(kind, axis, base, r, h):
    c = Cylinder(axis, base, r, h)
    q, w = c.quadrature(kind)

    def f(x):
        return sc.dot(axis / h, x - base)

    v = (f(q) * w).sum()
    assert_allclose(v, r**2 * h * np.pi / 2, atol=sc.scalar(1e-12))
