from functools import partial
from typing import Any, Literal

import scipp as sc

from .material import Material
from .types import SampleShape


def compute_transmission_map(
    sample_shape: SampleShape,
    sample_material: Material,
    beam_direction: sc.Variable,
    wavelength: sc.Variable,
    detector_position: sc.Variable,
    quadrature_kind: Literal['cheap', 'medium', 'expensive'] | Any = 'medium',
) -> sc.DataArray:
    """
    Computes transmission probability of single-scattered neutrons.

    Computes the probability that a neutron is transmitted to
    ``detector_position`` given that it travelled in ``beam_direction`` and
    scattered incoherently a single time when passing through the sample.

    .. math::
      C(\\mathbf{p}, \\lambda) = \\int_{Sample} \\exp{(-\\mu(\\lambda)
      L(\\mathbf{p}, \\mathbf{x}))} \\ d\\mathbf{x}

    where :math:`L` is the length of the path through the sample,
    :math:`\\mu` is the material dependent attenuation factor,
    and :math:`\\mathbf{p}` is the ``detector_position``.

    Parameters
    ----------
    sample_shape:
        The size and shape of the sample.
    sample_material:
        The sample material, this parameter determines the
        absorption and scattering coefficients.
    beam_direction:
        The direction of the incoming beam.
    wavelength:
        An array of wavelengths for which to evaluate the transmission fraction.
    detector_position:
        An array of vectors representing the scattering directions
        where the transmission fraction is evaluated.
    quadrature_kind:
        What kind of quadrature to use.
        A denser quadrature makes the result more accurate but takes longer to compute.
        What options exists depend on the sample shape.

    Returns
    -------
    :
        the transmission fraction as a function of detector_position and wavelength

    """
    points, weights = sample_shape.quadrature(quadrature_kind)
    transmission = _integrate_transmission_fraction(
        partial(
            _single_scatter_distance_through_sample,
            sample_shape,
            points,
            beam_direction,
        ),
        partial(_transmission_fraction, sample_material),
        points,
        weights,
        detector_position,
        wavelength,
    )
    return sc.DataArray(
        data=transmission / sample_shape.volume,
        coords={'detector_position': detector_position, 'wavelength': wavelength},
    )


def _single_scatter_distance_through_sample(
    sample_shape, scatter_point, initial_direction, scatter_direction
):
    L1 = sample_shape.beam_intersection(scatter_point, -initial_direction)
    L2 = sample_shape.beam_intersection(scatter_point, scatter_direction)
    return L1 + L2


def _transmission_fraction(material, distance_through_sample, wavelength):
    return sc.exp(
        -(material.attenuation_coefficient(wavelength) * distance_through_sample).to(
            unit='dimensionless', copy=False
        )
    )


def _integrate_transmission_fraction(
    distance_through_sample,
    transmission,
    points,
    weights,
    detector_position,
    wavelengths,
):
    # If size after broadcast is too large
    # then don't vectorize the operation to avoid OOM
    if points.size * detector_position.size > 20_000_000:
        out = []
        dim = detector_position.dims[0]
        for i in range(detector_position.sizes[dim]):
            out.append(  # noqa: PERF401
                _integrate_transmission_fraction(
                    distance_through_sample,
                    transmission,
                    points,
                    weights,
                    detector_position[dim, i],
                    wavelengths,
                )
            )

        return sc.concat(
            out,
            dim=dim,
        )

    scatter_direction = detector_position - points.to(unit=detector_position.unit)
    scatter_direction /= sc.norm(scatter_direction)
    Ltot = distance_through_sample(scatter_direction)

    # The Ltot array is already large, to avoid OOM, don't vectorize this operation
    return sc.concat(
        [
            # Instead of broadcast multiply and sum, use matvec for efficiency
            # this becomes relevant when the number of wavelength points grows
            sc.array(
                dims=(tf := transmission(Ltot, w)).dims[:-1],
                values=tf.values @ weights.values,
                unit=tf.unit * weights.unit,
            )
            for w in wavelengths
        ],
        dim=wavelengths.dim,
    )
