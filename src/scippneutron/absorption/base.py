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
    btheta = sc.broadcast(theta, sizes={**phi.sizes, **theta.sizes})
    bphi = sc.broadcast(phi, sizes={**phi.sizes, **theta.sizes})
    scatter_directions = sc.vectors(
        dims=bphi.dims,
        values=sc.concat(
            [
                sc.sin(btheta) * sc.cos(bphi),
                sc.sin(btheta) * sc.sin(bphi),
                sc.cos(btheta),
            ],
            dim='row',
        )
        .transpose([*bphi.dims, 'row'])
        .values,
    )
    Ltot = single_scatter_distance_through_sample(
        sample_shape, points, beam_direction, scatter_directions
    )
    total_transmission = sc.concat(
        # The Ltot array is already large, to avoid OOM, don't vectorize this operation
        [
            (transmission(sample_material, Ltot, w) * weights).sum(weights.dim)
            / sample_shape.volume
            for w in wavelength
        ],
        dim=wavelength.dim,
    )
    return sc.DataArray(
        data=total_transmission,
        coords={'phi': phi, 'theta': theta, 'wavelength': wavelength},
    )
