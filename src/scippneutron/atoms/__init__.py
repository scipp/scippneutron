# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Parameters of atoms and neutron interactions with atoms."""

from __future__ import annotations

import dataclasses
import importlib.resources
import re
from functools import lru_cache
from typing import TextIO

import scipp as sc


@dataclasses.dataclass(frozen=True, eq=False)
class Atom:
    """Atomic parameters of a specific element / isotope.

    Values have been retrieved at 2025-03-20T15:11 from the lists at
    https://www.ciaaw.org/atomic-weights.htm

    Atomic weights are properties of an *element* while atomic masses
    are properties of a specific *isotope* (nuclide).
    The reported atomic weights are the abridged standard
    atomic weight for each element.
    The reported atomic masses are the most recent value for each isotope
    (at the above date).
    """

    isotope: str
    z: int | None
    _atomic_weight: sc.Variable | None
    _atomic_mass: sc.Variable | None

    @property
    def atomic_weight(self) -> sc.Variable:
        """Return the atomic weight is available."""
        if self._atomic_weight is None:
            raise ValueError(
                f"Atomic weight for '{self.isotope}' is not defined ."
                "This likely means that there is no standard atomic weight "
                "for this element."
            )
        return self._atomic_weight.copy()

    @property
    def atomic_mass(self) -> sc.Variable:
        """Return the atomic mass is available."""
        if self._atomic_mass is None:
            raise ValueError(
                f"Atomic mass for '{self.isotope}' is not defined ."
                "This likely means that you specified an element name, not a specific "
                "isotope; atomic masses are only defined for isotopes / nuclides."
            )
        return self._atomic_mass.copy()

    def __eq__(self, other: object) -> bool | type(NotImplemented):
        if not isinstance(other, Atom):
            return NotImplemented
        return all(
            _eq_or_identical(getattr(self, field.name), getattr(other, field.name))
            for field in dataclasses.fields(self)
        )

    @staticmethod
    @lru_cache
    def for_isotope(isotope: str) -> Atom:
        """Return the atom parameters for the given element / isotope.

        Parameters
        ----------
        isotope:
            Name of the element or isotope.
            For example, 'H', '3He', 'V', '50V'.

        Returns
        -------
        :
            Atom parameters.
        """
        element = _parse_isotope_name(isotope)
        z, weight = _load_atomic_weight(element)
        if element == isotope:
            mass = None  # masses are only defined for specific isotopes
        else:
            mass = _load_atomic_mass(isotope)
        return Atom(
            isotope=isotope,
            z=z,
            _atomic_weight=weight,
            _atomic_mass=mass,
        )


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
    coherent_scattering_length_re: sc.Variable | None = None
    """Bound coherent scattering length (real part)."""
    coherent_scattering_length_im: sc.Variable | None = None
    """Bound coherent scattering length (imaginary part)."""
    incoherent_scattering_length_re: sc.Variable | None = None
    """Bound incoherent scattering length (real part)."""
    incoherent_scattering_length_im: sc.Variable | None = None
    """Bound incoherent scattering length (imaginary part)."""
    coherent_scattering_cross_section: sc.Variable | None = None
    """Bound coherent scattering cross-section."""
    incoherent_scattering_cross_section: sc.Variable | None = None
    """Bound incoherent scattering cross-section."""
    total_scattering_cross_section: sc.Variable | None = None
    """Total bound scattering cross-section."""
    absorption_cross_section: sc.Variable | None = None
    """Absorption cross-section for λ = 1.7982 Å neutrons."""

    def __eq__(self, other: object) -> bool | type(NotImplemented):
        if not isinstance(other, ScatteringParams):
            return NotImplemented
        return all(
            _eq_or_identical(getattr(self, field.name), getattr(other, field.name))
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
        with _open_bundled_parameters_file('scattering_parameters.csv') as f:
            if line_remainder := _find_line_with_isotope(isotope, f):
                return ScatteringParams._parse_line(isotope, line_remainder)
        raise ValueError(f"No entry for element / isotope '{isotope}'")

    @staticmethod
    def _parse_line(isotope: str, line: str) -> ScatteringParams:
        line = line.rstrip().split(',')
        return ScatteringParams(
            isotope=isotope,
            coherent_scattering_length_re=_assemble_scalar(line[0], line[1], 'fm'),
            coherent_scattering_length_im=_assemble_scalar(line[2], line[3], 'fm'),
            incoherent_scattering_length_re=_assemble_scalar(line[4], line[5], 'fm'),
            incoherent_scattering_length_im=_assemble_scalar(line[6], line[7], 'fm'),
            coherent_scattering_cross_section=_assemble_scalar(
                line[8], line[9], 'barn'
            ),
            incoherent_scattering_cross_section=_assemble_scalar(
                line[10], line[11], 'barn'
            ),
            total_scattering_cross_section=_assemble_scalar(line[12], line[13], 'barn'),
            absorption_cross_section=_assemble_scalar(line[14], line[15], 'barn'),
        )


def _open_bundled_parameters_file(name: str) -> TextIO:
    return importlib.resources.files('scippneutron.atoms').joinpath(name).open('r')


def _find_line_with_isotope(isotope: str, io: TextIO) -> str | None:
    while line := io.readline():
        name, rest = line.split(',', 1)
        if name == isotope:
            return rest
    return None


def _load_atomic_weight(element: str) -> tuple[int, sc.Variable | None]:
    # The CSV file was extracted from https://www.ciaaw.org/abridged-atomic-weights.htm
    # using the notebook in tools/atomic_weights.ipynb (in the ScippNeutron repo).
    with _open_bundled_parameters_file('atomic_weights.csv') as f:
        f.readline()  # skip copyright
        f.readline()  # skip header
        if line_remainder := _find_line_with_isotope(element, f):
            z, weight, error = line_remainder.rstrip().split(',')
            return int(z), _assemble_scalar(weight, error, 'Da')
    raise ValueError(f"No entry for element '{element}'")


def _load_atomic_mass(isotope: str) -> sc.Variable | None:
    # The CSV file was extracted from https://www.ciaaw.org/atomic-masses.htm
    # using the notebook in tools/atomic_weights.ipynb (in the ScippNeutron repo).
    with _open_bundled_parameters_file('atomic_masses.csv') as f:
        f.readline()  # skip copyright
        f.readline()  # skip header
        if line_remainder := _find_line_with_isotope(isotope, f):
            weight, error = line_remainder.rstrip().split(',')
            return _assemble_scalar(weight, error, 'Da')
    raise ValueError(f"No entry for element / isotope '{isotope}'")


def _assemble_scalar(value: str, std: str, unit: str) -> sc.Variable | None:
    if not value:
        return None
    value = float(value)
    variance = float(std) ** 2 if std else None
    return sc.scalar(value, variance=variance, unit=unit)


def _eq_or_identical(a: object, b: object) -> bool:
    if isinstance(a, sc.Variable) or isinstance(b, sc.Variable):
        return sc.identical(a, b)
    return a == b


def _parse_isotope_name(name: str) -> str:
    # Extract the element name from an isotope name.
    # 'H' -> 'H'
    # '2H' -> 'H'
    return re.match(r'(?:\d+)?([a-zA-Z]+)', name)[1]
