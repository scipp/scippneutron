import scipp as sc


def single_scatter_distance_through_sample(
    sample_shape, scatter_point, initial_direction, scatter_direction
):
    L1 = sample_shape.beam_intersection(scatter_point, -initial_direction)
    L2 = sample_shape.beam_intersection(scatter_point, scatter_direction)
    return L1 + L2


def transmission_fraction(material, distance_through_sample, wavelength):
    return sc.exp(
        -(material.attenuation_coefficient(wavelength) * distance_through_sample).to(
            unit='dimensionless'
        )
    )


def compute_transmission_map(
    sample_shape,
    sample_material,
    beam_direction,
    wavelength,
    detector_position,
    quadrature_kind='expensive',
):
    points, weights = sample_shape.quadrature(quadrature_kind)
    scatter_direction = detector_position - points.to(unit=detector_position.unit)
    scatter_direction /= sc.norm(scatter_direction)

    Ltot = single_scatter_distance_through_sample(
        sample_shape, points, beam_direction, scatter_direction
    )
    total_transmission = sc.concat(
        # The Ltot array is already large, to avoid OOM, don't vectorize this operation
        [
            (transmission_fraction(sample_material, Ltot, w) * weights).sum(weights.dim)
            / sample_shape.volume
            for w in wavelength
        ],
        dim=wavelength.dim,
    )
    return sc.DataArray(
        data=total_transmission,
        coords={'detector_position': detector_position, 'wavelength': wavelength},
    )
