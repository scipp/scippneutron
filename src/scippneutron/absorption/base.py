import scipp as sc


def single_scatter_distance_through_sample(
    sample_shape, scatter_point, initial_direction, scatter_direction
):
    L1 = sample_shape.beam_intersection(scatter_point, -initial_direction)
    L2 = sample_shape.beam_intersection(scatter_point, scatter_direction)
    return L1 + L2


def transmission(sample_material, distance_through_sample, wavelength):
    return sample_material.c * sc.exp(
        -sample_material.mu(wavelength) * distance_through_sample
    )


def compute_transmission_map(
    sample_shape,
    sample_material,
    beam_direction,
    wavelength,
    theta,
    phi,
    quadrature_kind='expensive',
):
    points, weights = sample_shape.quadrature(quadrature_kind)
    scatter_directions = sc.vectors(
        dims=['theta', 'phi'],
        values=sc.concat(
            [
                sc.sin(theta) * sc.cos(phi),
                sc.sin(theta) * sc.sin(phi),
                sc.broadcast(sc.cos(theta), sizes={**phi.sizes, **theta.sizes}),
            ],
            dim='row',
        )
        .transpose(['theta', 'phi', 'row'])
        .values,
    )
    Ltot = single_scatter_distance_through_sample(
        sample_shape, points, beam_direction, scatter_directions
    )
    return sc.concat(
        # The Ltot array is already large, to avoid OOM, don't vectorize this operation
        [
            (transmission(sample_material, Ltot, w) * weights).sum(weights.dim)
            for w in wavelength
        ],
        dim=wavelength.dim,
    )
