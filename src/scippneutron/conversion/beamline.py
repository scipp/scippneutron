# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
r"""Functions for computing coordinates related to beamline geometry.

Most functions in this module assume a straight beamline geometry.
That is, the beams are not curved by, e.g., a beam guide.
Specialized functions are provided for handling gravity.

ScippNeutron uses three positions to define a beamline:

- ``source_position`` defines the point where :math:`t=0` (or vice versa).
  In practice, this can be the actual neutron source, a moderator, or a chopper.
- ``sample_position`` is the position of the sample that scatters neutrons.
- ``position`` is the position of the detector (pixel / voxel) that detects a neutron
  at :math:`t=\mathsf{tof}`.

Based on these positions, we define:

- ``incident_beam`` is the vector of incoming neutrons on the sample.
  It points from the source to the sample.
  (:func:`straight_incident_beam`)
- ``L1`` (or :math:`L_1`) is the length of ``incident_beam``.
  (:func:`L1`)
- ``scattered_beam`` is the vector of neutrons that were scattered off the sample.
  It points from the sample to the detector pixels.
  (:func:`straight_scattered_beam`)
- ``L2`` (or :math:`L_2`) is the length of ``scattered_beam``.
  (:func:`L2`)
- ``Ltotal`` is the total beam length :math:`L_\mathsf{total} = L_1 + L_2`.
  (:func:`total_beam_length` and :func:`total_straight_beam_length_no_scatter`)

Coordinate system
-----------------

Note
----
  The coordinate system is not needed by most operations because beamline coordinates
  are usually relative to the positions.
  As long as ``position``, ``source_position``, and ``sample_position`` are defined
  consistently, ``incident_beam``, ``scattered_beam``, and ``two_theta`` can be
  computed without knowledge of the coordinate system.

  However, when we need ``phi``, or need to correct ``two_theta`` for gravity,
  we need the actual coordinate system.

ScippNeutron uses a coordinate system aligned with
the incident beam and gravity.
ScippNeutron's coordinate system corresponds to that of
`NeXus <https://manual.nexusformat.org/design.html#the-nexus-coordinate-system>`_.
The image below shows how coordinates are defined with respect to the
quantities defined above.
The plot on the right-hand side shows the view from the sample towards the source,
that is, the :math:`z`-axis points towards the viewer.
(The sample is placed in the origin here; this is only done for illustration purposes
and not required.)

.. image:: ../../../docs/_static/beamline/beamline_coordinates_light.svg
   :class: only-light
   :scale: 100 %
   :alt: Beamline coordinate system.
   :align: center

.. image:: ../../../docs/_static/beamline/beamline_coordinates_dark.svg
   :class: only-dark
   :scale: 100 %
   :alt: Beamline coordinate system.
   :align: center

The axes are defined by these unit vectors:

.. math::

    \hat{e}_z &= b_1 / |b_1| \\
    \hat{e}_y &= -g / |g| \\
    \hat{e}_x &= \hat{e}_y \times \hat{e}_z

which span an orthogonal, right-handed coordinate system.
Here, :math:`b_1` is the ``incident_beam`` and :math:`g` is the gravity vector.
This means that the z-axis is parallel to the incident beam and the y-axis is
antiparallel to gravity.
Gravity must be orthogonal to the incident beam for this definition to produce
and orthogonal coordinate system.
Basis vectors can be computed using :func:`beam_aligned_unit_vectors`.

:math:`p = \sqrt{x^2 + y^2}` is the projection of the
scattered beam onto the :math:`x-y` plane.
It is included here because it is used by some coordinate transformations.

The scattering angles are defined similarly to spherical coordinates as:

.. math ::

    \mathsf{cos}(2\theta) &= \frac{b_1 \cdot b_2}{|b_1| |b_2|} \\
    \mathsf{tan}(\phi) &= \frac{b_2 \cdot \hat{e}_y}{b_2 \cdot \hat{e}_x}

where :math:`b_2` is the scattered beam.

- ``two_theta`` is the angle between the scattered beam and incident beam.
  Note the extra factor 2 compared to spherical coordinates;
  it ensures that the definition corresponds to Bragg's law.
- ``phi`` is the angle between the x-axis and the projection of the scattered beam
  onto the x-y-plane.

These definitions assume that gravity can be neglected.
See :func:`scattering_angles_with_gravity` for definitions that account for gravity.
And :func:`scattering_angle_in_yz_plane` for the definition used in reflectometry ---
which also includes gravity.
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
    straight_incident_beam:
        Compute the incident beam for a straight beamline.
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
    straight_scattered_beam:
        Compute the scattered beam for a straight beamline.
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

    See :mod:`beamline` for the definition of the angle.

    The result is equivalent to

    .. math::

        b_1 &= \\mathtt{incident\\_beam} / |\\mathtt{incident\\_beam}| \\\\
        b_2 &= \\mathtt{scattered\\_beam} / |\\mathtt{scattered\\_beam}| \\\\
        2\\theta &= \\mathsf{acos}(b_1 \\cdot b_2)

    but uses a numerically more stable implementation by W. Kahan
    described in paragraph of :cite:`kahan:2000:CPR`.
    See also
    https://people.eecs.berkeley.edu/~wkahan/MathH110/Cross.pdf.

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
    straight_incident_beam:
        Compute the incident beam for a straight beamline.
    straight_scattered_beam:
        Compute the scattered beam for a straight beamline.
    scattering_angles_with_gravity:
        Calculate ``two_theta`` and ``phi`` with gravity.
    """
    # The implementation is based on paragraph 13 of
    # https://people.eecs.berkeley.edu/~wkahan/MathH110/Cross.pdf
    # It claims that the formula is 'valid for euclidean spaces of any dimension'
    # and 'it never errs by more than a modest multiple of epsilon'
    # where 'epsilon is the roundoff threshold for individual arithmetic
    # operations'.
    b1 = incident_beam / L1(incident_beam=incident_beam)
    b2 = scattered_beam / L2(scattered_beam=scattered_beam)
    return 2 * sc.atan2(y=sc.norm(b1 - b2), x=sc.norm(b1 + b2))


class BeamAlignedUnitVectors(TypedDict):
    """A dict with keys 'beam_aligned_unit_{x,y,z}'."""

    beam_aligned_unit_x: sc.Variable
    """Unit vector in x-direction."""
    beam_aligned_unit_y: sc.Variable
    """Unit vector in y-direction."""
    beam_aligned_unit_z: sc.Variable
    """Unit vector in z-direction."""


def beam_aligned_unit_vectors(
    incident_beam: sc.Variable, gravity: sc.Variable
) -> BeamAlignedUnitVectors:
    r"""Return unit vectors for a coordinate system aligned with the incident beam.

    The unit vectors are

    .. math::

        \hat{e}_z &= b_1 / |b_1| \\
        \hat{e}_y &= -g / |g| \\
        \hat{e}_x &= \hat{e}_y \times \hat{e}_z

    where :math:`b_1` is the ``incident_beam`` and :math:`g` is the gravity vector.
    See the module-level docs of :mod:`scippneutron.conversion.beamline` for  details.

    Parameters
    ----------
    incident_beam:
        Beam from source to sample. Expects ``dtype=vector3``.
    gravity:
        Gravity vector. Expects ``dtype=vector3``.

    Returns
    -------
    A dict containing the unit vectors with keys 'ex', 'ey', and 'ez'.

    See Also
    --------
    straight_incident_beam:
        Compute the incident beam for a straight beamline.
    """
    if sc.any(
        abs(sc.dot(gravity, incident_beam))
        > sc.scalar(1e-10, unit=incident_beam.unit) * sc.norm(gravity)
    ):
        raise ValueError(
            '`gravity` and `incident_beam` must be orthogonal. '
            f'Got a deviation of {sc.dot(gravity, incident_beam).max():c}. '
            'This is required to fully define spherical coordinates theta and phi.'
        )

    ez = incident_beam / sc.norm(incident_beam)
    ey = -gravity / sc.norm(gravity)
    ex = sc.cross(ey, ez)
    return {
        'beam_aligned_unit_x': ex,
        'beam_aligned_unit_y': ey,
        'beam_aligned_unit_z': ez,
    }


def _drop_due_to_gravity(
    distance: sc.Variable,
    wavelength: sc.Variable,
    gravity: sc.Variable,
) -> sc.Variable:
    """Compute the distance a neutron drops due to gravity.

    See the documentation of :func:`scattering_angles_with_gravity`.
    """
    distance = distance.to(dtype=elem_dtype(wavelength), copy=False)
    const = (sc.norm(gravity) * (sc.constants.m_n**2 / (2 * sc.constants.h**2))).to(
        dtype=elem_dtype(wavelength), copy=False
    )

    # Convert unit to eventually match the unit of y.
    # Copy to make it safe to use in-place ops.
    drop = wavelength.to(
        unit=sc.sqrt(sc.reciprocal(elem_unit(distance) * elem_unit(const))), copy=True
    )
    drop *= drop
    drop *= const

    distance *= distance
    if set(distance.dims).issubset(drop.dims):
        drop *= distance
        return drop
    return drop * distance


class SphericalCoordinates(TypedDict):
    """A dict with keys 'two_theta' and 'phi'."""

    two_theta: sc.Variable
    """Polar angle.

    Between the scattered beam and continuation of the incident beam.
    Mind the factor 2 which is defined as in Bragg's law.
    """

    phi: sc.Variable
    """Azimuthal angle.

    The angle between the projection of the scattered beam onto the x-y plane
    and the x-axis.
    """


def scattering_angles_with_gravity(
    incident_beam: sc.Variable,
    scattered_beam: sc.Variable,
    wavelength: sc.Variable,
    gravity: sc.Variable,
) -> SphericalCoordinates:
    r"""Compute scattering angles theta and phi using gravity.

    With the definitions of the unit vectors in
    `Coordinate system <./scippneutron.conversion.beamline.rst#coordinate-system>`_,
    we have the components of the scattered beam :math:`b_2` in the beam-aligned
    coordinate system:

    .. math::

        x_d &= b_2 \cdot \hat{e}_x \\
        y_d &= b_2 \cdot \hat{e}_y \\
        z_d &= b_2 \cdot \hat{e}_z

    :math:`b_2` points from the sample to the detector that detected the neutron.
    The neutron left the sample in the direction of a vector :math:`b'_2`.
    This vector defines the scattering angles :math:`2\theta` and :math:`\phi`.
    Taking gravity into account, we have :math:`b'_2 \neq b_2`.
    Solving the equations of motion gives the following for the
    components of :math:`b'_2`:

    .. math::

        x'_d &= x_d \\
        y'_d &= y_d + \frac{|g| m_n^2}{2 h^2} L_2^{\prime\, 2} \lambda^2 \\
        z'_d &= z_d

    Where :math:`|g|` is the strength of gravity, :math:`m_n` is the neutron mass,
    :math:`h` is the Planck constant, and :math:`\lambda` is the wavelength.
    This gives the gravity-corrected scattering angles:

    .. math::

        \mathsf{tan}(2\theta) &= \frac{\sqrt{x_d^2 + y_d^{\prime\, 2}}}{z_d} \\
        \mathsf{tan}(\phi) &= \frac{y'_d}{x_d}

    Attention
    ---------
        The above equation for :math:`y'_d` contains :math:`L_2^{\prime\, 2} = |b'_2|`
        which in turn depends on :math:`y'_d`.
        Solving this equation for :math:`y'_d` is too difficult.
        Instead, we approximate :math:`L'_2 \approx L_2`.
        The impact of this approximation on :math:`2\theta` is of the order of
        :math:`10^{-6}` or less for beamlines at ESS.
        This is within the expected statistical uncertainties and can be ignored.

        See `two_theta gravity correction
        <../../user-guide/algorithms-background/two_theta-gravity-correction.rst>`_
        for details.

    Parameters
    ----------
    incident_beam:
        Beam from source to sample. Expects ``dtype=vector3``.
    scattered_beam:
        Beam from sample to detector. Expects ``dtype=vector3``.
    wavelength:
        Wavelength of neutrons.
    gravity:
        Gravity vector.

    Returns
    -------
    :
        A dict containing the polar scattering angle ``'two_theta'`` and
        the azimuthal angle ``'phi'``.

    See also
    --------
    scattering_angle_in_yz_plane:
        Ignores the ``x`` component when computing ``theta``.
        This is used in reflectometry.
    """
    unit_vectors = beam_aligned_unit_vectors(
        incident_beam=incident_beam, gravity=gravity
    )
    ex = unit_vectors['beam_aligned_unit_x']
    ey = unit_vectors['beam_aligned_unit_y']
    ez = unit_vectors['beam_aligned_unit_z']

    y = _drop_due_to_gravity(
        distance=sc.norm(scattered_beam), wavelength=wavelength, gravity=gravity
    )
    y += sc.dot(scattered_beam, ey).to(dtype=elem_dtype(wavelength), copy=False)

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


def scattering_angle_in_yz_plane(
    incident_beam: sc.Variable,
    scattered_beam: sc.Variable,
    wavelength: sc.Variable,
    gravity: sc.Variable,
) -> sc.Variable:
    r"""Compute polar scattering angles in the y-z plane using gravity.

    Note
    ----
        This function uses the reflectometry definition of the polar scattering angle.
        Other techniques define the angle w.r.t. the incident beam.
        See :func:`scattering_angles_with_gravity` for those use cases.

    With the definitions given in :func:`scattering_angles_with_gravity`,
    and ignoring :math:`x_d`, we get

    .. math::

        \mathsf{tan}(\gamma) = \frac{|y_d^{\prime}|}{z_d}

    with

    .. math::

        y'_d = y_d + \frac{|g| m_n^2}{2 h^2} L_2^{\prime\, 2} \lambda^2

    The angle :math:`\gamma` is defined as in Fig. 5 of :cite:`STAHN201644`.

    Attention
    ---------
        The above equation for :math:`y'_d` contains :math:`L_2^{\prime\, 2} = |b'_2|`
        which in turn depends on :math:`y'_d`.
        Solving this equation for :math:`y'_d` is too difficult.
        Instead, we approximate :math:`L'_2 \approx L_2`.
        The impact of this approximation on :math:`\gamma` is of the order of
        :math:`10^{-6}` or less for beamlines at ESS.
        This is within the expected statistical uncertainties and can be ignored.

        See `two_theta gravity correction
        <../../user-guide/algorithms-background/two_theta-gravity-correction.rst>`_
        for details.

    Parameters
    ----------
    incident_beam:
        Beam from source to sample. Expects ``dtype=vector3``.
    scattered_beam:
        Beam from sample to detector. Expects ``dtype=vector3``.
    wavelength:
        Wavelength of neutrons.
    gravity:
        Gravity vector.

    Returns
    -------
    :
        The polar scattering angle :math:`\gamma`.

    See also
    --------
    scattering_angles_with_gravity:
        Includes the ``x`` component when computing ``theta``.
        This is used in techniques other than reflectometry.
    """
    unit_vectors = beam_aligned_unit_vectors(
        incident_beam=incident_beam, gravity=gravity
    )
    ey = unit_vectors['beam_aligned_unit_y']
    ez = unit_vectors['beam_aligned_unit_z']

    y = _drop_due_to_gravity(
        distance=sc.norm(scattered_beam), wavelength=wavelength, gravity=gravity
    )
    y += sc.dot(scattered_beam, ey).to(dtype=elem_dtype(wavelength), copy=False)
    y = sc.abs(y, out=y)
    z = sc.dot(scattered_beam, ez).to(dtype=elem_dtype(y), copy=False)
    return sc.atan2(y=y, x=z, out=y)
