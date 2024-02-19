# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Parameters for neutron interactions with atoms."""

import importlib.resources
from functools import lru_cache
from typing import Optional, TextIO

import scipp as sc


def _open_scattering_parameters_file() -> TextIO:
    return (
        importlib.resources.files('scippneutron.atoms')
        .joinpath('scattering_parameters.csv')
        .open('r')
    )


@lru_cache()
def scattering_params(isotope: str) -> dict[str, Optional[sc.Variable]]:
    """Return the scattering parameters for the given element / isotope.

    Provides access to the scattering lengths and cross-sections of neutrons
    with a given element or isotope.
    Values have been retrieved at 2024-02-19T17:00:00Z from the list at
    https://www.ncnr.nist.gov/resources/n-lengths/list.html
    which is based on :cite:`sears:1992`.

    Parameters
    ----------
    isotope:
        Name of the element or isotope.
        For example, 'H', '3He', 'V', '50V'.

    Returns
    -------
    :
        Dict with the scattering parameters.
        A value can be ``None`` if the parameter is not available.
        Scattering lengths are split into real and imaginary parts.

        Keys:

            - ``bound_coherent_scattering_length_re``
            - ``bound_coherent_scattering_length_im``
            - ``bound_incoherent_scattering_length_re``
            - ``bound_incoherent_scattering_length_im``
            - ``bound_coherent_scattering_cross_section``
            - ``bound_incoherent_scattering_cross_section``
            - ``total_bound_scattering_cross_section``
            - ``absorption_cross_section``
    """
    with _open_scattering_parameters_file() as f:
        while line := f.readline():
            name, rest = line.split(',', 1)
            if name == isotope:
                return _parse_line(rest)
    raise ValueError(f"No entry for element / isotope '{isotope}'")


def _parse_line(line: str) -> dict[str, Optional[sc.Variable]]:
    line = line.rstrip().split(',')
    return {
        'bound_coherent_scattering_length_re': _assemble_scalar(line[0], line[1], 'fm'),
        'bound_coherent_scattering_length_im': _assemble_scalar(line[2], line[3], 'fm'),
        'bound_incoherent_scattering_length_re': _assemble_scalar(
            line[4], line[5], 'fm'
        ),
        'bound_incoherent_scattering_length_im': _assemble_scalar(
            line[6], line[7], 'fm'
        ),
        'bound_coherent_scattering_cross_section': _assemble_scalar(
            line[8], line[9], 'barn'
        ),
        'bound_incoherent_scattering_cross_section': _assemble_scalar(
            line[10], line[11], 'barn'
        ),
        'total_bound_scattering_cross_section': _assemble_scalar(
            line[12], line[13], 'barn'
        ),
        'absorption_cross_section': _assemble_scalar(line[14], line[15], 'barn'),
    }


def _assemble_scalar(value: str, std: str, unit: str) -> Optional[sc.Variable]:
    if not value:
        return None
    value = float(value)
    variance = float(std) ** 2 if std else None
    return sc.scalar(value, variance=variance, unit=unit)
