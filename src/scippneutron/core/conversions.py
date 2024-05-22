# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

from collections.abc import Callable

import scipp as sc

from ..conversion import graph as _graphs


def _inelastic_scatter_graph(energy_mode):
    inelastic_graph_factory = {
        'direct_inelastic': _graphs.tof.direct_inelastic,
        'indirect_inelastic': _graphs.tof.indirect_inelastic,
    }
    return {
        **_graphs.beamline.beamline(scatter=True),
        **inelastic_graph_factory[energy_mode](start='tof'),
    }


def _reachable_by(target, graph):
    return any(
        target == targets if isinstance(targets, str) else target in targets
        for targets in graph.keys()
    )


def _elastic_scatter_graph(origin, target):
    scatter_graph_kinematics = _graphs.beamline.beamline(scatter=True)
    if _reachable_by(target, scatter_graph_kinematics):
        return dict(scatter_graph_kinematics)
    return {**scatter_graph_kinematics, **_graphs.tof.elastic(origin)}


def _scatter_graph(origin, target, energy_mode):
    graph = (
        _elastic_scatter_graph(origin, target)
        if energy_mode == 'elastic'
        else _inelastic_scatter_graph(energy_mode)
    )
    return graph


def conversion_graph(
    origin: str, target: str, scatter: bool, energy_mode: str
) -> dict[str | tuple[str], Callable]:
    """
    Get a conversion graph for given parameters.

    The graph can be used with `scipp.transform_coords`.

    :param origin: Name of the input coordinate.
    :param target: Name of the output coordinate.
    :param scatter: Choose whether to use scattering or non-scattering conversions.
    :param energy_mode: Select if energy is conserved. One of `'elastic'`,
                        `'direct_inelastic'`, `'indirect_inelastic'`.
    :return: Conversion graph.
    :seealso: :py:func:`scippneutron.convert`,
              :py:func:`scippneutron.deduce_conversion_graph`.
    """

    # Results are copied to ensure users do not modify the global dictionaries.
    if scatter:
        return dict(_scatter_graph(origin, target, energy_mode))
    else:
        return {
            **_graphs.beamline.beamline(scatter=False),
            **_graphs.tof.kinematic(start='tof'),
        }


def _find_inelastic_inputs(data):
    return [name for name in ('incident_energy', 'final_energy') if name in data.coords]


def _deduce_energy_mode(data, origin, target):
    inelastic_inputs = _find_inelastic_inputs(data)
    if target == 'energy_transfer':
        if len(inelastic_inputs) > 1:
            raise RuntimeError(
                "Data contains coords for incident *and* final energy, cannot have "
                "both for inelastic scattering."
            )
        if len(inelastic_inputs) == 0:
            raise RuntimeError(
                "Data contains neither coords for incident nor for final energy, this "
                "does not appear to be inelastic-scattering data, cannot convert to "
                "energy transfer."
            )
        return {
            'incident_energy': 'direct_inelastic',
            'final_energy': 'indirect_inelastic',
        }[inelastic_inputs[0]]

    if 'energy' in (origin, target):
        if inelastic_inputs:
            raise RuntimeError(
                f"Data contains coords for inelastic scattering "
                f"({inelastic_inputs}) but conversion with elastic energy requested. "
                f"This is not implemented."
            )
    return 'elastic'


def deduce_conversion_graph(
    data: sc.DataArray | sc.Dataset, origin: str, target: str, scatter: bool
) -> dict[str | tuple[str], Callable]:
    """
    Get the conversion graph used by :py:func:`scippneutron.convert`
    when called with identical arguments.

    :param data: Input data.
    :param origin: Name of the input coordinate.
    :param target: Name of the output coordinate.
    :param scatter: Choose whether to use scattering or non-scattering conversions.
    :return: Conversion graph.
    :seealso: :py:func:`scippneutron.convert`, :py:func:`scippneutron.conversion_graph`.
    """
    return conversion_graph(
        origin, target, scatter, _deduce_energy_mode(data, origin, target)
    )


def convert(
    data: sc.DataArray | sc.Dataset, origin: str, target: str, scatter: bool
) -> sc.DataArray | sc.Dataset:
    """
    Perform a unit conversion from the given origin unit to target.
    See the documentation page on "Coordinate Transformations"
    (https://scipp.github.io/scippneutron/user-guide/coordinate-transformations.html)
    for more details.

    :param data: Input data.
    :param origin: Name of the input coordinate.
    :param target: Name of the output coordinate.
    :param scatter: Choose whether to use scattering or non-scattering conversions.
    :return: A new scipp.DataArray or scipp.Dataset with the new coordinate.
    :seealso: :py:func:`scippneutron.deduce_conversion_graph` and
              :py:func:`scippneutron.conversion_graph` to inspect
              the possible conversions.
    """

    graph = deduce_conversion_graph(data, origin, target, scatter)

    try:
        converted = data.transform_coords(target, graph=graph)
    except KeyError as err:
        if err.args[0] == target:
            raise RuntimeError(
                f"No viable conversion from '{origin}' to '{target}' "
                f"with scatter={scatter}."
            ) from None
        raise RuntimeError(
            f"Missing coordinate '{err.args[0]}' for conversion "
            f"from '{origin}' to '{target}'"
        ) from None

    return converted
