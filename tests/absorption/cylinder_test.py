import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_allclose

from scippneutron.absorption.cylinder import Cylinder


@pytest.fixture(
    params=[
        sc.vector([-0.9854498123542775, -0.1699666653521188, 0]),
        sc.vector([0.7242293076582138, 0.6895592142295717, 0]),
        sc.vector([0.9445601257691059, 0.3283384972966938, 0]),
        sc.vector([-0.8107703958381735, -0.5853643012965615, 0]),
        sc.vector([0.48126033718549477, -0.8765777135269319, 0]),
    ]
)
def point_on_unit_circle(request):
    return request.param


@pytest.fixture(params=[sc.scalar(1.2), sc.scalar(0.2)])
def height(request):
    return request.param


@pytest.fixture(params=[sc.scalar(1.2), sc.scalar(0.2)])
def radius(request):
    return request.param


@pytest.fixture(
    params=[
        sc.vector([0.0, 0.0, 1.0]),
        sc.vector([-0.5620126808026259, -0.1798933776079791, 0.8073290031392648]),
    ]
)
def axis(request):
    return request.param


@pytest.fixture(
    params=[
        sc.vector([0.0, 0.0, 0.0]),
        sc.vector([1.0, -2.0, 3.0]),
    ]
)
def base(request):
    return request.param


@pytest.fixture
def cylinder(request, axis, base, radius, height):
    return Cylinder(axis, base, radius, height)


def _rotate_from_z_to_axis(p, ax):
    z = sc.vector([0, 0, 1.0])
    if ax == z:
        return p
    u = sc.cross(z, ax)
    un = sc.norm(u)
    u *= sc.asin(un) / un
    return sc.spatial.rotations_from_rotvecs(u) * p


def test_intersection_in_base(cylinder, point_on_unit_circle):
    v = _rotate_from_z_to_axis(point_on_unit_circle, cylinder.symmetry_line)
    assert_allclose(
        cylinder.beam_intersection(
            cylinder.radius * v
            + cylinder.center_of_base
            # Move start point just inside cylinder.
            # Required to make all tests pass.
            + 2 * np.finfo(float).eps * cylinder.symmetry_line,
            -v,
        ),
        2 * cylinder.radius,
    )


def test_intersection_diagonal(cylinder, point_on_unit_circle):
    v = _rotate_from_z_to_axis(point_on_unit_circle, cylinder.symmetry_line)
    n = -2 * cylinder.radius * v + cylinder.symmetry_line * cylinder.height
    n /= sc.norm(n)
    # Diagonal intersection is as expected
    assert_allclose(
        cylinder.beam_intersection(cylinder.radius * v + cylinder.center_of_base, n),
        ((2 * cylinder.radius) ** 2 + cylinder.height**2) ** 0.5,
    )
    # Intersection is zero in other directions
    assert_allclose(
        cylinder.beam_intersection(cylinder.radius * v + cylinder.center_of_base, -n),
        sc.scalar(0.0),
        atol=sc.scalar(1e-14),
    )
    n = 2 * cylinder.radius * v + cylinder.symmetry_line * cylinder.height
    n /= sc.norm(n)
    assert_allclose(
        cylinder.beam_intersection(cylinder.radius * v + cylinder.center_of_base, n),
        sc.scalar(0.0),
        atol=sc.scalar(1e-14),
    )
    assert_allclose(
        cylinder.beam_intersection(cylinder.radius * v + cylinder.center_of_base, -n),
        sc.scalar(0.0),
        atol=sc.scalar(1e-14),
    )


def test_intersection_along_axis(cylinder, point_on_unit_circle):
    v = _rotate_from_z_to_axis(point_on_unit_circle, cylinder.symmetry_line)
    # Move point just inside cylinder
    x = cylinder.radius * v + cylinder.center_of_base - 2 * np.finfo(float).eps * v
    assert_allclose(
        cylinder.beam_intersection(x, cylinder.symmetry_line), cylinder.height
    )


def test_no_intersection(cylinder, point_on_unit_circle):
    v = _rotate_from_z_to_axis(point_on_unit_circle, cylinder.symmetry_line)
    x = cylinder.center_of_base + cylinder.radius * v
    assert_allclose(
        cylinder.beam_intersection(x, v), sc.scalar(0.0), atol=sc.scalar(1e-14)
    )
    # Rotate 90 deg around cylinder axis.
    # Should still not intersect the cylinder.
    v = (
        sc.spatial.rotations_from_rotvecs(
            cylinder.symmetry_line * sc.scalar(np.pi / 2, unit='rad')
        )
        * v
    )
    assert_allclose(
        cylinder.beam_intersection(x, v), sc.scalar(0.0), atol=sc.scalar(1e-7)
    )


def test_intersection_from_center(cylinder, point_on_unit_circle):
    v = _rotate_from_z_to_axis(point_on_unit_circle, cylinder.symmetry_line)
    assert_allclose(cylinder.beam_intersection(cylinder.center, v), cylinder.radius)
    assert_allclose(
        cylinder.beam_intersection(cylinder.center, cylinder.symmetry_line),
        cylinder.height / 2,
    )
    assert_allclose(
        cylinder.beam_intersection(cylinder.center, -cylinder.symmetry_line),
        cylinder.height / 2,
    )


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
            # Exact value from Sympy
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


@pytest.mark.parametrize('kind', ['mc', ('mc', 4000)])
def test_quadrature_mc(kind):
    import numpy

    numpy.random.seed(1)

    c = Cylinder(
        sc.vector([0, 0, 1.0]), sc.vector([0, 0, 0.0]), sc.scalar(1.0), sc.scalar(1.0)
    )

    def f(x):
        return sc.sin(sc.dot(x, sc.vector([1, 0.5, 0]) * sc.scalar(1, unit='rad'))) ** 2

    expected = sc.scalar(0.7972445081889596)
    q, w = c.quadrature(kind)
    v = (f(q) * w).sum()
    assert_allclose(v, expected, atol=sc.scalar(5e-2), rtol=sc.scalar(5e-2))

    if isinstance(kind, tuple):
        assert len(q) == kind[1]


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
