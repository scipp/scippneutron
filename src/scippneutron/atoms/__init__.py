# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Parameters for neutron interactions with atoms."""

from __future__ import annotations

import dataclasses
import importlib.resources
from functools import lru_cache
from typing import TextIO

import scipp as sc


def reference_wavelength() -> sc.Variable:
    """Return the reference wavelength for absorption cross-sections.

    Returns
    -------
    :
        1.7982 Å
    """
    return sc.scalar(1.7982, unit='angstrom')


@dataclasses.dataclass(frozen=True, eq=False)
class ScatteringParams:
    """Scattering parameters for neutrons with a specific element / isotope.

    Provides access to the scattering lengths and cross-sections of neutrons
    with a given element or isotope.
    Values have been retrieved at 2024-02-19T17:00:00Z from the list at
    https://www.ncnr.nist.gov/resources/n-lengths/list.html
    which is based on :cite:`sears:1992`.
    Values are ``None`` where the table does not provide values.

    The absorption cross-section applies to neutrons with a wavelength
    of 1.7982 Å.
    See :func:`reference_wavelength`.
    """

    isotope: str
    """Element / isotope name."""
    coherent_scattering_length_re: sc.Variable | None
    """Bound coherent scattering length (real part)."""
    coherent_scattering_length_im: sc.Variable | None
    """Bound coherent scattering length (imaginary part)."""
    incoherent_scattering_length_re: sc.Variable | None
    """Bound incoherent scattering length (real part)."""
    incoherent_scattering_length_im: sc.Variable | None
    """Bound incoherent scattering length (imaginary part)."""
    coherent_scattering_cross_section: sc.Variable | None
    """Bound coherent scattering cross-section."""
    incoherent_scattering_cross_section: sc.Variable | None
    """Bound incoherent scattering cross-section."""
    total_scattering_cross_section: sc.Variable | None
    """Total bound scattering cross-section."""
    absorption_cross_section: sc.Variable | None
    """Absorption cross-section for λ = 1.7982 Å neutrons."""

    def __eq__(self, other: object) -> bool | type(NotImplemented):
        if not isinstance(other, ScatteringParams):
            return NotImplemented
        return all(
            self.isotope == other.isotope
            if field.name == 'isotope'
            else _eq_or_identical(getattr(self, field.name), getattr(other, field.name))
            for field in dataclasses.fields(self)
        )

    @staticmethod
    @lru_cache
    def for_isotope(isotope: str) -> ScatteringParams:
        """Return the scattering parameters for the given element / isotope.

        Parameters
        ----------
        isotope:
            Name of the element or isotope.
            For example, 'H', '3He', 'V', '50V'.

        Returns
        -------
        :
            Neutron scattering parameters.
        """
        with _open_scattering_parameters_file() as f:
            while line := f.readline():
                name, rest = line.split(',', 1)
                if name == isotope:
                    return _parse_line(isotope, rest)
        raise ValueError(f"No entry for element / isotope '{isotope}'")


def _open_scattering_parameters_file() -> TextIO:
    return (
        importlib.resources.files('scippneutron.atoms')
        .joinpath('scattering_parameters.csv')
        .open('r')
    )


def _parse_line(isotope: str, line: str) -> ScatteringParams:
    line = line.rstrip().split(',')
    return ScatteringParams(
        isotope=isotope,
        coherent_scattering_length_re=_assemble_scalar(line[0], line[1], 'fm'),
        coherent_scattering_length_im=_assemble_scalar(line[2], line[3], 'fm'),
        incoherent_scattering_length_re=_assemble_scalar(line[4], line[5], 'fm'),
        incoherent_scattering_length_im=_assemble_scalar(line[6], line[7], 'fm'),
        coherent_scattering_cross_section=_assemble_scalar(line[8], line[9], 'barn'),
        incoherent_scattering_cross_section=_assemble_scalar(
            line[10], line[11], 'barn'
        ),
        total_scattering_cross_section=_assemble_scalar(line[12], line[13], 'barn'),
        absorption_cross_section=_assemble_scalar(line[14], line[15], 'barn'),
    )


def _assemble_scalar(value: str, std: str, unit: str) -> sc.Variable | None:
    if not value:
        return None
    value = float(value)
    variance = float(std) ** 2 if std else None
    return sc.scalar(value, variance=variance, unit=unit)


def _eq_or_identical(a: sc.Variable | None, b: sc.Variable | None) -> bool:
    if a is None:
        return b is None
    return sc.identical(a, b)
