# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
"""
Functions for computing coordinates related to beamline geometry.

Most functions in this module assume a straight beamline geometry.
That is, the beams are not curved by, e.g., a beam guide.
All quantities are defined as in the image below:

.. image:: ../../../docs/_static/straight-beamline.svg
   :scale: 75 %
   :alt: Scattering angle in Bragg reflection.
   :align: center

The 'source position' defines the point where :math:`t=0` (or vice versa).
Neutrons are scattered at the 'sample position' and detected
at 'position' at :math:`t=\\mathsf{tof}`.

In addition, ``total_straight_beam_length_no_scatter`` can be used if the beam
does not scatter off of a sample.
This is useful for instance for monitors.
In this case, ``Ltotal`` is the distance from source to detector.
"""

from typing import TypedDict

import scipp as sc
import scipp.constants
from scipp.typing import VariableLike

from .._utils import elem_dtype, elem_unit


def L1(*, incident_beam: VariableLike) -> VariableLike:
    """Compute the length of the incident beam.

    The result is the primary beam length

    .. math::

        L_1 = | \\mathtt{incident\\_beam} |

    Parameters
    ----------
    incident_beam:
        Beam from source to sample. Expects ``dtype=vector3``.

    Returns
    -------
    :
        :math:`L_1`

    See Also
    --------
    scippneutron.conversions.beamline.straight_incident_beam:
    """
    return sc.norm(incident_beam)


def L2(*, scattered_beam: VariableLike) -> VariableLike:
    """Compute the length of the scattered beam.

    The result is the secondary beam length

    .. math::

        L_2 = | \\mathtt{incident\\_beam} |

    Parameters
    ----------
    scattered_beam:
        Beam from sample to detector. Expects ``dtype=vector3``.

    Returns
    -------
    :
        :math:`L_2`

    See Also
    --------
    scippneutron.conversions.beamline.straight_scattered_beam:
    """
    return sc.norm(scattered_beam)


def straight_incident_beam(
    *, source_position: VariableLike, sample_position: VariableLike
) -> VariableLike:
    """Compute the length of the beam from source to sample.

    Assumes a straight beam.
    The result is

    .. math::

        \\mathtt{incident\\_beam} = \\mathtt{sample\\_position}
                                    - \\mathtt{source\\_position}

    Parameters
    ----------
    source_position:
        Position of the beam's source.
    sample_position:
        Position of the sample.

    Returns
    -------
    :
        ``incident_beam``
    """
    return sample_position - source_position


def straight_scattered_beam(
    *, position: VariableLike, sample_position: VariableLike
) -> VariableLike:
    """Compute the length of the beam from sample to detector.

    Assumes a straight beam.
    The result is

    .. math::

        \\mathtt{scattered\\_beam} = \\mathtt{position} - \\mathtt{sample\\_position}

    Parameters
    ----------
    position:
        Position of the detector.
    sample_position:
        Position of the sample.

    Returns
    -------
    :
        ``scattered_beam``
    """
    return position - sample_position


def total_beam_length(*, L1: VariableLike, L2: VariableLike) -> VariableLike:
    """Compute the combined length of the incident and scattered beams.

    The result is

    .. math::

        L_\\mathsf{total} = | L_1 + L_2 |

    Parameters
    ----------
    L1:
        Primary path length (incident beam).
    L2:
        Secondary path length (scattered beam).

    Returns
    -------
    :
        :math:`L_\\mathsf{total}`
    """
    return L1 + L2


def total_straight_beam_length_no_scatter(
    *, source_position: VariableLike, position: VariableLike
) -> VariableLike:
    """Compute the length of the beam from source to given position.

    Assumes a straight beam.
    The result is

    .. math::

        L_\\mathsf{total} = | \\mathtt{position} - \\mathtt{{source\\_position}} |

    Parameters
    ----------
    source_position:
        Position of the beam's source. Expects ``dtype=vector3``.
    position:
        Position of the detector. Expects ``dtype=vector3``.

    Returns
    -------
    :
        :math:`L_\\mathsf{total}`
    """
    return sc.norm(position - source_position)


def two_theta(
    *, incident_beam: VariableLike, scattered_beam: VariableLike
) -> VariableLike:
    """Compute the scattering angle between scattered and transmitted beams.

    See :mod:`scippneutron.conversions.beamline` for the definition of the angle.

    The result is equivalent to

    .. math::

        b_1 &= \\mathtt{incident\\_beam} / |\\mathtt{incident\\_beam}| \\\\
        b_2 &= \\mathtt{scattered\\_beam} / |\\mathtt{scattered\\_beam}| \\\\
        2\\theta &= \\mathsf{acos}(b_1 \\cdot b_2)

    but uses a numerically more stable implementation by W. Kahan
    (https://people.eecs.berkeley.edu/~wkahan/MathH110/Cross.pdf).

    Parameters
    ----------
    incident_beam:
        Beam from source to sample. Expects ``dtype=vector3``.
    scattered_beam:
        Beam from sample to detector. Expects ``dtype=vector3``.

    Returns
    -------
    :
        :math:`2\\theta`

    See Also
    --------
    scippneutron.conversions.beamline.straight_incident_beam:
    scippneutron.conversions.beamline.straight_scattered_beam:
    """
    # TODO use proper citation
    # The implementation is based on paragraph 13 of
    # https://people.eecs.berkeley.edu/~wkahan/MathH110/Cross.pdf
    # Which is Kahan:2000:CPR in
    # https://netlib.org/bibnet/authors/k/kahan-william-m.html
    # And referenced by https://scicomp.stackexchange.com/a/27769
    #
    # It claims that the formula is 'valid for euclidean spaces of any dimension'
    # and 'it never errs by more than a modest multiple of epsilon'
    # where 'epsilon is the roundoff threshold for individual arithmetic
    # operations'.
    b1 = incident_beam / L1(incident_beam=incident_beam)
    b2 = scattered_beam / L2(scattered_beam=scattered_beam)
    return 2 * sc.atan2(y=sc.norm(b1 - b2), x=sc.norm(b1 + b2))


class SphericalCoordinates(TypedDict):
    two_theta: sc.Variable
    phi: sc.Variable


# TODO clean up
# TODO document coordinate system
# TODO document definition of angles -> images + equations
# TODO check numerical error, compare with vector based calculation
#   not sure which one is more precise but I suspect it's the vector based one
# TODO remove helper functions or keep?
#   If keep, need to clearly document coord system and rename


# def scattering_angles_with_gravity(
#     incident_beam: sc.Variable,
#     scattered_beam: sc.Variable,
#     wavelength: sc.Variable,
#     gravity: sc.Variable,
# ) -> SphericalCoordinates:
#     grav = sc.norm(gravity)
#
#     if sc.any(
#         abs(sc.dot(gravity, incident_beam))
#         > sc.scalar(1e-10, unit=incident_beam.unit) * grav
#     ):
#         raise ValueError(
#             '`gravity` and `incident_beam` must be orthogonal. '
#             f'Got a deviation of {sc.dot(gravity, scattered_beam).max():c}. '
#             'This is required to fully define spherical coordinates theta and phi.'
#         )
#
#     L2 = sc.norm(scattered_beam)
#
#     x_term = cylindrical_x(cyl_x_unit_vector(gravity, incident_beam), scattered_beam)
#
#     y_term = sc.to_unit(wavelength, elem_unit(L2), copy=True)
#     y_term *= y_term
#     drop = L2**2
#     drop *= grav * (sc.constants.m_n**2 / (2 * sc.constants.h**2))
#     # Optimization when handling either the dense or the event coord of binned data:
#     # - For the event coord, both operands have same dims, and we can
#     #   multiply in place
#     # - For the dense coord, we need to broadcast using non in-place operation
#     if set(drop.dims).issubset(set(y_term.dims)):
#         y_term *= drop
#     else:
#         y_term = drop * y_term
#     y_term += cylindrical_y(cyl_y_unit_vector(gravity), scattered_beam)
#     phi = sc.atan2(y=y_term, x=x_term)
#
#     x_term *= x_term
#     y_term *= y_term
#
#     if set(x_term.dims).issubset(set(y_term.dims)):
#         y_term += x_term
#     else:
#         y_term = y_term + x_term
#     out = sc.sqrt(y_term, out=y_term)
#     out /= L2
#     out = sc.asin(out, out=out)
#     return {'two_theta': out, 'phi': phi}


def _beam_aligned_unit_vectors(
    reference_beam: sc.Variable, up: sc.Variable
) -> tuple[sc.Variable, sc.Variable, sc.Variable]:
    ez = reference_beam / sc.norm(reference_beam)
    ey = up / sc.norm(up)
    ex = sc.cross(ey, ez)
    return ex, ey, ez


def scattering_angles_with_gravity(
    incident_beam: sc.Variable,
    scattered_beam: sc.Variable,
    wavelength: sc.Variable,
    gravity: sc.Variable,
) -> SphericalCoordinates:
    g = sc.norm(gravity)

    if sc.any(
        abs(sc.dot(gravity, incident_beam))
        > sc.scalar(1e-10, unit=incident_beam.unit) * g
    ):
        raise ValueError(
            '`gravity` and `incident_beam` must be orthogonal. '
            f'Got a deviation of {sc.dot(gravity, scattered_beam).max():c}. '
            'This is required to fully define spherical coordinates theta and phi.'
        )

    scattered_beam = scattered_beam
    ex, ey, ez = _beam_aligned_unit_vectors(reference_beam=incident_beam, up=-gravity)
    y = sc.dot(scattered_beam, ey).to(dtype=elem_dtype(wavelength), copy=False)

    L2 = sc.norm(scattered_beam).to(dtype=elem_dtype(wavelength), copy=False)
    factor = (g * (sc.constants.m_n**2 / (2 * sc.constants.h**2))).to(
        dtype=elem_dtype(wavelength), copy=False
    )

    # Convert unit to eventually match the unit of y.
    # Copy to make it safe to use in-place ops.
    a = wavelength.to(
        unit=sc.sqrt(sc.reciprocal(elem_unit(L2) * elem_unit(factor))), copy=True
    )
    a *= a
    a *= factor

    drop = L2
    drop *= drop
    drop = drop * a  # TODO in-place when possible
    del a

    y = y + drop  # TODO in-place when possible
    del drop, L2

    x = sc.dot(scattered_beam, ex).to(dtype=elem_dtype(y), copy=False)
    phi = sc.atan2(y=y, x=x)

    # Corresponds to `two_theta_ = sc.atan2(y=sc.sqrt(x**2 + y**2), x=z)`
    x *= x
    y *= y
    y += x
    del x
    y = sc.sqrt(y, out=y)
    z = sc.dot(scattered_beam, ez).to(dtype=elem_dtype(y), copy=False)
    two_theta_ = sc.atan2(y=y, x=z, out=y)

    return {'two_theta': two_theta_, 'phi': phi}
