# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""CIF file writer.

This module contains tools for writing `CIF <https://www.iucr.org/resources/cif>`_
files with diffraction data.
`It does not support reading CIF files.`

CIF Builder: High-level interface
---------------------------------
This module supports two interfaces for writing files.
The high-level interface uses a builder pattern for assembling files.
It is implemented by :class:`CIF`.

To demonstrate the interface, first make some mockup powder diffraction data:

  >>> import scipp as sc
  >>> tof = sc.array(dims=['tof'], values=[1.2, 1.4, 2.3], unit='us')
  >>> intensity = sc.array(
  ...     dims=['tof'],
  ...     values=[13.6, 26.0, 9.7],
  ...     variances=[0.7, 1.1, 0.5],
  ... )
  >>> da = sc.DataArray(intensity, coords={'tof': tof})

Assemble data and metadata using a :class:`CIF` builder:

  >>> from scippneutron.io import cif
  >>> from scippneutron import metadata
  >>> cif_ = (
  ...  cif.CIF('my-data', comment="This is a demo of ScippNeutron's CIF builder.")
  ...  .with_beamline(metadata.Beamline(name='fake', facility='made up'))
  ...  .with_authors(metadata.Person(
  ...      name='Jane Doe',
  ...      orcid_id='0000-0000-0000-0001',
  ...      corresponding=True
  ...  ))
  ...  .with_reduced_powder_data(da)
  ... )

When all data has been added, write it to a file:

  >>> cif_.save('example.cif')

This results in a file containing

.. code-block:: text

    #\\#CIF_1.1
    # This is a demo of ScippNeutron's CIF builder.
    data_my-data

    loop_
    _audit_conform.dict_name
    _audit_conform.dict_version
    _audit_conform.dict_location
    pdCIF 2.5.0 https://github.com/COMCIFS/Powder_Dictionary/blob/7608b92165f58f968f054344e67662e01d4b401a/cif_pow.dic
    coreCIF 3.3.0 https://github.com/COMCIFS/cif_core/blob/fc3d75a298fd7c0c3cde43633f2a8616e826bfd5/cif_core.dic

    _audit.creation_date 2024-09-05T13:47:54+00:00
    _audit.creation_method 'Written by scippneutron v24.6.1

    _audit_contact_author.name 'Jane Doe'
    _audit_contact_author.id_orcid 0000-0000-0000-0001

    _diffrn_source.beamline fake
    _diffrn_source.facility 'made up'

    loop_
    _pd_meas.time_of_flight
    _pd_proc.intensity_net
    _pd_proc.intensity_net_su
    1.2 13.6 0.8366600265340756
    1.4 26.0 1.0488088481701516
    2.3 9.7 0.7071067811865476

Chunks and loops: Low-level interface
-------------------------------------
The high-level CIF builder uses :class:`Block`, :class:`Chunk`, and :class:`Loop` to
encode data.
Those classes can also be used directly to gain more control over the file contents.

To demonstrate, make mockup powder diffraction data:

  >>> import scipp as sc
  >>> tof = sc.array(dims=['tof'], values=[1.2, 1.4, 2.3], unit='us')
  >>> intensity = sc.array(
  ...     dims=['tof'],
  ...     values=[13.6, 26.0, 9.7],
  ...     variances=[0.7, 1.1, 0.5],
  ... )

Wrap the data in a ``Loop`` to write them together as columns.

  >>> from scippneutron.io import cif
  >>> tof_loop = cif.Loop({
  ...     'pd_meas.time_of_flight': tof,
  ...     'pd_meas.intensity_total': sc.values(intensity),
  ...     'pd_meas.intensity_total_su': sc.stddevs(intensity),
  ... })

Write the data to file along with some metadata:

  >>> block = cif.Block('example', [
  ...     {
  ...         'diffrn_radiation.probe': 'neutron',
  ...         'diffrn_source.beamline': 'some-beamline',
  ...     },
  ...     tof_loop,
  ... ])
  >>> cif.save_cif('example.cif', block)

This results in a file containing

.. code-block::

  #\\#CIF_1.1
  data_example

  _diffrn_radiation.probe neutron
  _diffrn_source.beamline some-beamline

  loop_
  _pd_meas.time_of_flight
  _pd_meas.intensity_total
  _pd_meas.intensity_total_su
  1.2 13.6 0.8366600265340756
  1.4 26.0 1.0488088481701516
  2.3 9.7 0.7071067811865476
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Mapping
from copy import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import scipp as sc

from ..metadata import Beamline, Person, Source, SourceType
from ._files import open_or_pass


@dataclass(frozen=True)
class CIFSchema:
    name: str
    version: str
    location: str


CORE_SCHEMA = CIFSchema(
    name='coreCIF',
    version='3.3.0',
    location='https://github.com/COMCIFS/cif_core/blob/6f8502e81b623eb0fd779c79efaf191d49fa198c/cif_core.dic',
)
PD_SCHEMA = CIFSchema(
    name='pdCIF',
    version='2.5.0',
    location='https://github.com/COMCIFS/Powder_Dictionary/blob/970c2b2850a923796db5f4e9b7207d10ab5fd8e5/cif_pow.dic',
)


def save_cif(
    fname: str | Path | TextIO,
    content: Block | Iterable[Block] | CIF,
    *,
    comment: str = '',
) -> None:
    """Save data blocks to a CIF file.

    To use, first create :class:`scippneutron.io.cif.Block` objects to collect and
    structure data for the file, then use this function to write the file.

    Parameters
    ----------
    fname:
        Path or file handle for the output file.
    content:
        One or more CIF data blocks or a ``CIF`` object to write to the file.
    comment:
        Optional comment that can be written at the top of the file.
    """
    if isinstance(content, CIF):
        if comment:
            content = copy(content)
            content.comment = comment
        content.save(fname)
        return

    if isinstance(content, Block):
        content = (content,)
    with open_or_pass(fname, "w") as f:
        _write_file_heading(f, comment=comment)
        _write_multi(f, content)


class CIF:
    """A builder for CIF files.

    This class implements a builder pattern for defining the contents of a CIF file.
    See the module documentation of :mod:`cif` for examples.
    """

    def __init__(self, name='', *, comment: str = '') -> None:
        self._block = Block(
            name=name
        )  # We mainly keep this around for managing the name.

        self._comment = ''
        self.comment = comment
        # Keep a separate list from self._block to assemble items
        # in a specific order when saving.
        self._content: list[Chunk | Loop] = []
        self._authors: list[Person] = []
        self._reducers: list[str] = []

        # Should be long enough to never run out of IDs.
        self._id_generator = (str(i) for i in range(1, 1_000_000_000))

    @property
    def name(self) -> str:
        return self._block.name

    @name.setter
    def name(self, name: str) -> None:
        self._block.name = name

    @property
    def comment(self) -> str:
        """Optional comment that can be written at the top of the file."""
        return self._comment

    @comment.setter
    def comment(self, comment: str) -> None:
        self._comment = _encode_non_ascii(comment)

    @property
    def schema(self) -> set[CIFSchema]:
        """CIF schemas used for the file."""
        return self._block.schema

    def save(self, fname: str | Path | TextIO) -> None:
        """
        Parameters
        ----------
        fname:
            Path or file handle for the output file.
        """
        block = self._block.copy()
        _add_audit(block, self._reducers)
        for item in (*self._assemble_authors(), *self._content):
            block.add(item)
        save_cif(fname, block, comment=self._comment)

    def copy(self) -> CIF:
        """Return a copy of this builder."""
        cif_ = CIF(name=self.name, comment=self.comment)
        cif_._block = self._block.copy()
        cif_._content.extend(self._content)
        cif_._authors.extend(self._authors)
        cif_._reducers.extend(self._reducers)
        return cif_

    def with_beamline(
        self,
        beamline: Beamline,
        source: Source | None = None,
        *,
        comment: str = '',
    ) -> CIF:
        """Add beamline information."""
        device = _get_beamline_device(beamline, source)
        probe = _get_beamline_probe(beamline, source)
        fields = {
            'diffrn_radiation.probe': probe,
            'diffrn_source.beamline': beamline.name,
            'diffrn_source.facility': beamline.facility,
            'diffrn_source.device': device,
        }

        cif_ = self.copy()
        cif_._content.append(
            Chunk(
                {k: v for k, v in fields.items() if v is not None},
                comment=comment,
                schema=CORE_SCHEMA,
            )
        )
        return cif_

    def with_reduced_powder_data(self, data: sc.DataArray, *, comment: str = '') -> CIF:
        """Add a loop with reduced powder data.

        The input must be 1-dimensional with a dimension name in
        ``('tof', 'dspacing')``.
        The data array may also have a name in
        ``('intensity_net', 'intensity_norm', 'intensity_total')``.
        If the name is not set, it defaults to ``'intensity_norm'``,
        which is appropriate for typical outputs from data reduction.
        See https://github.com/COMCIFS/Powder_Dictionary/blob/master/cif_pow.dic

        The data gets written as intensity along a single coord whose
        name matches the dimension name.
        Standard uncertainties are also written if present.

        The unit of the coordinate must match the requirement of pdCIF.

        Parameters
        ----------
        data:
            1-dimensional data array with a recognized dimension name
        comment:
            Optional comment that can be written above the data in the file.

        Returns
        -------
        :
            A builder with added reduced data.

        Examples
        --------
        Make mockup powder diffraction data:

          >>> import scipp as sc
          >>> tof = sc.array(dims=['tof'], values=[1.2, 1.4, 2.3], unit='us')
          >>> intensity = sc.array(
          ...     dims=['tof'],
          ...     values=[13.6, 26.0, 9.7],
          ...     variances=[0.7, 1.1, 0.5],
          ... )

        Add to a CIF builder:

          >>> from scippneutron.io import cif
          >>> cif_ = cif.CIF('reduced-data')
          >>> da = sc.DataArray(intensity, coords={'tof': tof})
          >>> cif_ = cif_.with_reduced_powder_data(da)
        """
        cif_ = self.copy()
        cif_._content.append(_make_reduced_powder_loop(data, comment=comment))
        return cif_

    def with_powder_calibration(self, cal: sc.DataArray, *, comment: str = '') -> CIF:
        r"""Add a powder calibration table.

        The calibration data encode the following transformation from
        d-spacing to time-of-flight:

        .. math::

            t = \sum_{i=0}^N\, c_i d^{p_i}

        where :math:`c_i` is the i-th element of ``cal`` and :math:`p^{p_i}`
        is the i-th element of ``cal.coords['power']``.

        Parameters
        ----------
        cal:
            The data are the calibration coefficients (possibly with variances).
            Must have a coordinate called ``'power'`` defining :math:`p` in the
            equation above.
        comment:
            Optional comment that can be written above the data in the file.

        Returns
        -------
        :
            A builder with added calibration data.

        Examples
        --------
        Add a mockup calibration table:

          >>> import scipp as sc
          >>> from scippneutron.io import cif
          >>> cal = sc.DataArray(
          ...     sc.array(dims=['cal'], values=[3.4, 0.2]),
          ...     coords={'power': sc.array(dims=['cal'], values=[0, 1])},
          ... )
          >>> cif_ = cif.CIF('powder-calibration')
          >>> cif_ = cif_.with_powder_calibration(cal)
        """
        cif_ = self.copy()
        cif_._content.append(_make_powder_calibration_loop(cal, comment=comment))
        return cif_

    def with_authors(self, *authors: Person) -> CIF:
        """Add one or more authors.

        Parameters
        ----------
        authors:
            Authors to add to this CIF object.

        Returns
        -------
        :
            A builder with added calibration data.
        """
        cif_ = self.copy()
        cif_._authors.extend(authors)
        return cif_

    def with_reducers(self, *reducers: str) -> CIF:
        """Add one or more programs that were used to reduce the software.

        Parameters
        ----------
        reducers:
            Pieces of software that were used to reduce the data.
            Each string is one piece of software in freeform notation.
            It is recommended to include the program name and version.

        Returns
        -------
        :
            A builder with added calibration data.
        """
        cif_ = self.copy()
        cif_._reducers.extend(reducers)
        return cif_

    def _assemble_authors(self) -> list[Chunk | Loop]:
        contact = [author for author in self._authors if author.corresponding]
        regular = [author for author in self._authors if not author.corresponding]

        results = []
        roles = {}
        for authors, category in zip(
            (contact, regular), ('audit_contact_author', 'audit_author'), strict=True
        ):
            if not authors:
                continue
            data, rols = _serialize_authors(authors, category, self._id_generator)
            results.append(data)
            roles.update(rols)
        if roles:
            results.append(_serialize_roles(roles))

        return results


class _CIFBase:
    def __init__(self, *, comment: str = '', schema: CIFSchema) -> None:
        self._comment = ''
        self.comment = comment  # use str-encoding logic
        self._schema = _preprocess_schema(schema)

    @property
    def comment(self) -> str:
        """Optional comment that can be written above the object in the file."""
        return self._comment

    @comment.setter
    def comment(self, comment: str) -> None:
        self._comment = _encode_non_ascii(comment)

    @property
    def schema(self) -> set[CIFSchema]:
        """CIF schemas used for the object."""
        return self._schema


class Chunk(_CIFBase):
    """A group of CIF key-value pairs.

    Chunks contain one or more key-value pairs where values are scalars,
    i.e., not arrays.
    Chunks are represented in files as a group of pairs separated from
    other chunks and loops by empty lines.

    Note that CIF has no concept of chunks; they are only used for organizing
    data in ScippNeutron.
    """

    def __init__(
        self,
        pairs: Mapping[str, Any] | Iterable[tuple[str, Any]] | None,
        /,
        *,
        comment: str = '',
        schema: CIFSchema | Iterable[CIFSchema] | None = None,
    ) -> None:
        """Create a new CIF chunk.

        Parameters
        ----------
        pairs:
            Defines a mapping from keys (a.k.a. tags) to values.
        comment:
            Optional comment that can be written above the chunk in the file.
        schema:
            CIF schemas used for the chunk.
            Content is not checked against the schema, but the schema is written
            to the file.
        """
        super().__init__(comment=comment, schema=schema)
        self._pairs = dict(pairs) if pairs is not None else {}

    def __setitem__(self, key: str, value: Any) -> None:
        """Add a key-value pair to the chunk."""
        self._pairs[key] = value

    def write(self, f: TextIO) -> None:
        """Write this chunk to a file.

        Used mainly internally, use :func:`scippneutron.io.cif.save_cif` instead.

        Parameters
        ----------
        f:
            File handle.
        """
        _write_comment(f, self.comment)
        for key, val in self._pairs.items():
            v = _format_value(val)
            if v.startswith(';'):
                f.write(f'_{key}\n{v}\n')
            else:
                f.write(f'_{key} {v}\n')


class Loop(_CIFBase):
    """A CIF loop.

    Contains a mapping from strings to Scipp variables.
    The strings are arbitrary and ``Loop`` can merge items from different categories
    into a single loop.
    All variables must have the same length.
    """

    def __init__(
        self,
        columns: Mapping[str, sc.Variable] | Iterable[tuple[str, sc.Variable]] | None,
        *,
        comment: str = '',
        schema: CIFSchema | Iterable[CIFSchema] | None = None,
    ) -> None:
        """Create a new CIF loop.

        Parameters
        ----------
        columns:
            Defines a mapping from column names (including their category)
            to column values as Scipp variables.
        comment:
            Optional comment that can be written above the loop in the file.
        schema:
            CIF schemas used for the loop.
            Content is not checked against the schema, but the schema is written
            to the file.
        """
        super().__init__(comment=comment, schema=schema)
        self._columns = {}
        for key, column in columns.items():
            self[key] = column

    def __setitem__(self, name: str, value: sc.Variable) -> None:
        """Add a column to the loop.

        Parameters
        ---------
        name:
            Column name.
        value:
            Values of the column.
        """
        if value.ndim != 1:
            raise sc.DimensionError(
                f"CIF loops can only contain 1d variables, got {value.ndim} dims"
            )
        if self._columns:
            existing = next(iter(self._columns.values())).sizes
            if existing != value.sizes:
                raise sc.DimensionError(
                    f"Inconsistent dims in CIF loop: {value.sizes} "
                    f"loop dims: {existing}"
                )

        self._columns[name] = value

    def write(self, f: TextIO) -> None:
        """Write this loop to a file.

        Used mainly internally, use :func:`scippneutron.io.cif.save_cif` instead.

        Parameters
        ----------
        f:
            File handle.
        """
        _write_comment(f, self.comment)
        f.write('loop_\n')
        for key in self._columns:
            f.write(f'_{key}\n')
        formatted_values = [
            tuple(map(_format_value, row))
            for row in zip(*self._columns.values(), strict=True)
        ]
        # If any value is a multi-line string, lay out elements as a flat vertical
        # list, otherwise use a 2d table.
        sep = (
            '\n'
            if any(';' in item for row in formatted_values for item in row)
            else ' '
        )
        for row in formatted_values:
            f.write(sep.join(row))
            f.write('\n')


class Block(_CIFBase):
    """A CIF data block.

    A block contains an ordered sequence of loops
    and chunks (groups of key-value-pairs).
    The contents are written to file in the order specified in the block.
    """

    def __init__(
        self,
        name: str,
        content: Iterable[Mapping[str, Any] | Loop | Chunk] | None = None,
        *,
        comment: str = '',
        schema: CIFSchema | Iterable[CIFSchema] | None = None,
    ) -> None:
        """Create a new CIF data block.

        Parameters
        ----------
        name:
            Name of the block.
            Can contain any non-whitespace characters.
            Can be at most 75 characters long.
        content:
            Initial loops and chunks.
            ``dicts`` are converted to :class:`scippneutron.io.cif.Chunk`.
        comment:
            Optional comment that can be written above the block in the file.
        schema:
            CIF schemas used for the block.
            Content is not checked against the schema, but the schema is written
            to the file.
        """
        super().__init__(comment=comment, schema=schema)
        self._name = ''
        self.name = name
        self._content = _convert_input_content(content) if content is not None else []

    @property
    def name(self) -> str:
        """Name of the block."""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = _encode_non_ascii(name)
        if ' ' in self._name or '\t' in self._name or '\n' in self._name:
            raise ValueError(
                "Block name must not contain spaces or line breaks, "
                f"got: '{self._name}'"
            )
        if len(self._name) > 75:
            warnings.warn(
                "cif.Block name should not be longer than 75 characters, got "
                f"{len(self._name)} characters ('{self._name}')",
                UserWarning,
                stacklevel=2,
            )

    @property
    def schema(self) -> set[CIFSchema]:
        """CIF schemas used for the block."""
        merged = set(super().schema)
        for item in self._content:
            merged.update(item.schema)
        return merged

    def copy(self) -> Block:
        """Return a shallow copy of the block."""
        return Block(
            self.name, list(self._content), comment=self.comment, schema=self.schema
        )

    def add(
        self,
        content: Mapping[str, Any] | Iterable[tuple[str, Any]] | Chunk | Loop,
        /,
        comment: str = '',
    ) -> None:
        """Add a chunk or loop to the end of the block.

        Parameters
        ----------
        content:
            A loop, chunk, or mapping to add.
            Mappings get converted to chunks.
        comment:
            Optional comment that can be written above the chunk or loop in the file.
        """
        if not isinstance(content, Chunk | Loop):
            content = Chunk(content, comment=comment)
        self._content.append(content)

    def write(self, f: TextIO) -> None:
        """Write this block to a file.

        Used mainly internally, use :func:`scippneutron.io.cif.save_cif` instead.

        Parameters
        ----------
        f:
            File handle.
        """
        schema_loop = _make_schema_loop(self.schema)

        _write_comment(f, self.comment)
        f.write(f'data_{self.name}\n\n')
        if schema_loop is not None:
            schema_loop.write(f)
            f.write('\n')
        _write_multi(f, self._content)


_KNOWN_SPALLATION_SOURCES = {
    'csns',
    'ess',
    'isis',
    'j-parc',
    'lanscesinq',
    'sns',
}


def _get_beamline_device(beamline: Beamline, source: Source | None) -> str | None:
    if source is None:
        if (beamline.facility or '').lower() in _KNOWN_SPALLATION_SOURCES:
            return 'spallation'
        else:
            return None
    match source.source_type:
        case SourceType.SpallationNeutronSource:
            return 'spallation'
        case SourceType.ReactorNeutronSource:
            return 'nuclear'
        case SourceType.SynchrotronXraySource:
            return 'synch'


def _get_beamline_probe(beamline: Beamline, source: Source | None) -> str | None:
    if source is None:
        if (beamline.facility or '').lower() in _KNOWN_SPALLATION_SOURCES:
            return 'neutron'
        else:
            return None
    match source.source_type:
        case SourceType.SpallationNeutronSource | SourceType.ReactorNeutronSource:
            return 'neutron'
        case SourceType.SynchrotronXraySource:
            return 'x-ray'


def _convert_input_content(
    content: Iterable[Mapping[str, Any] | Loop | Chunk],
) -> list[Loop | Chunk]:
    return [item if isinstance(item, Loop | Chunk) else Chunk(item) for item in content]


def _preprocess_schema(
    schema: CIFSchema | Iterable[CIFSchema] | None,
) -> set[CIFSchema]:
    if schema is None:
        return set()
    if isinstance(schema, CIFSchema):
        res = {schema}
    else:
        res = set(schema)
    res.add(CORE_SCHEMA)  # needed to encode schema itself
    return res


def _make_schema_loop(schema: set[CIFSchema]) -> Loop | None:
    if not schema:
        return None
    columns = {
        'audit_conform.dict_name': [],
        'audit_conform.dict_version': [],
        'audit_conform.dict_location': [],
    }
    for s in schema:
        columns['audit_conform.dict_name'].append(s.name)
        columns['audit_conform.dict_version'].append(s.version)
        columns['audit_conform.dict_location'].append(s.location)
    return Loop(
        {key: sc.array(dims=['schema'], values=val) for key, val in columns.items()}
    )


def _quotes_for_string_value(value: str) -> str | None:
    if '\n' in value:
        return ';'
    if "'" in value:
        if '"' in value:
            return ';'
        return '"'
    if '"' in value:
        return "'"
    if ' ' in value:
        return "'"
    if not value:
        return "'"  # so that empty strings are shown as ''
    return None


def _encode_non_ascii(s: str) -> str:
    return s.encode('ascii', 'backslashreplace').decode('ascii')


def _format_value(value: Any) -> str:
    if isinstance(value, sc.Variable):
        if value.variance is not None:
            without_unit = sc.scalar(value.value, variance=value.variance)
            s = f'{without_unit:c}'
        else:
            s = str(value.value)
    elif isinstance(value, datetime):
        s = value.isoformat()
    else:
        s = str(value)

    s = _encode_non_ascii(s)

    if (quotes := _quotes_for_string_value(s)) == ';':
        return f'; {s}\n;'
    elif quotes is not None:
        return quotes + s + quotes
    return s


def _write_comment(f: TextIO, comment: str) -> None:
    if comment:
        f.write('# ')
        f.write('\n# '.join(comment.splitlines()))
        f.write('\n')


def _write_multi(f: TextIO, to_write: Iterable[Any]) -> None:
    first = True
    for item in to_write:
        if not first:
            f.write('\n')
        first = False
        item.write(f)


def _write_file_heading(f: TextIO, comment: str) -> None:
    f.write('#\\#CIF_1.1\n')
    _write_comment(f, comment)


def _reduced_powder_coord(data) -> tuple[str, sc.Variable]:
    if data.ndim != 1:
        raise sc.DimensionError(f'Can only save 1d powder data, got {data.ndim} dims.')
    known_coords = {
        'tof': ('pd_meas.time_of_flight', 'us'),
        'dspacing': ('pd_proc.d_spacing', 'Å'),
    }
    try:
        name, unit = known_coords[data.dim]
    except KeyError:
        raise sc.CoordError(
            f'Unrecognized dim: {data.dim}. Must be one of {list(known_coords)}'
        ) from None

    coord = data.coords[data.dim]
    if coord.unit != unit:
        raise sc.UnitError(
            f'Incorrect unit for powder coordinate {name}: {coord.unit} expected {unit}'
        )
    return name, coord


def _normalize_reduced_powder_name(name: str) -> str:
    if name not in ('intensity_net', 'intensity_norm', 'intensity_total'):
        raise ValueError(f'Unrecognized name for reduced powder data: {name}')
    return f'pd_proc.{name}'


def _make_reduced_powder_loop(data: sc.DataArray, comment: str) -> Loop:
    coord_name, coord = _reduced_powder_coord(data)
    data_name = _normalize_reduced_powder_name(data.name or 'intensity_norm')

    res = Loop(
        {
            'pd_data.point_id': sc.arange(data.dim, len(data), unit=None),
            coord_name: sc.values(coord),
        },
        comment=comment,
        schema=PD_SCHEMA,
    )
    if coord.variances is not None:
        res[coord_name + '_su'] = sc.stddevs(coord)
    res[data_name] = sc.values(data.data)
    if data.variances is not None:
        res[data_name + '_su'] = sc.stddevs(data.data)

    if data.unit != 'one':
        pre = res.comment + '\n' if res.comment else ''
        res.comment = f'{pre}Unit of intensity: [{data.unit}]'

    return res


def _make_powder_calibration_loop(data: sc.DataArray, comment: str) -> Loop:
    # All names are valid python identifiers
    id_by_power = {0: 'ZERO', 1: 'DIFC', 2: 'DIFA', -1: 'DIFB'}
    ids = sc.array(
        dims=[data.dim],
        values=[
            id_by_power.get(power, f'c{power}'.replace('-', '_').replace('.', '_'))
            for power in data.coords['power'].values
        ],
    )
    res = Loop(
        {
            'pd_calib_d_to_tof.id': ids,
            'pd_calib_d_to_tof.power': data.coords['power'],
            'pd_calib_d_to_tof.coeff': sc.values(data.data),
        },
        comment=comment,
        schema=PD_SCHEMA,
    )
    if data.variances is not None:
        res['pd_calib_d_to_tof.coeff_su'] = sc.stddevs(data.data)
    return res


def _add_audit(block: Block, reducers: list[str]) -> None:
    from .. import __version__

    audit_chunk = Chunk(
        {
            'audit.creation_date': datetime.now(timezone.utc).replace(microsecond=0),
            'audit.creation_method': f'Written by scippneutron {__version__}',
        },
        schema=CORE_SCHEMA,
    )
    block.add(audit_chunk)

    if len(reducers) == 1:
        audit_chunk['computing.diffrn_reduction'] = reducers[0]
    elif len(reducers) > 1:
        block.add(
            Loop(
                {'computing.diffrn_reduction': sc.array(dims=['r'], values=reducers)},
                schema=CORE_SCHEMA,
            )
        )


def _serialize_authors(
    authors: list[Person],
    category: str,
    id_generator: Iterator[str],
) -> tuple[Chunk | Loop, dict[str, str]]:
    fields = {
        f'{category}.{key}': f
        for key in ('name', 'email', 'address', 'orcid_id')
        if any(f := [getattr(a, key) or '' for a in authors])
    }
    # Map between our name (Person.orcid_id) and CIF's (id_orcid)
    if orcid_id := fields.pop(f'{category}.orcid_id', None):
        fields[f'{category}.id_orcid'] = orcid_id

    roles = {next(id_generator): a.role for a in authors}
    if any(roles.values()):
        fields[f'{category}.id'] = list(roles.keys())
    roles = {key: val for key, val in roles.items() if val}

    if len(authors) == 1:
        return Chunk(
            {key: val[0] for key, val in fields.items()},
            schema=CORE_SCHEMA,
        ), roles
    return Loop(
        {key: sc.array(dims=['author'], values=val) for key, val in fields.items()},
        schema=CORE_SCHEMA,
    ), roles


def _serialize_roles(roles: dict[str, str]) -> Loop:
    return Loop(
        {
            'audit_author_role.id': sc.array(dims=['role'], values=list(roles)),
            'audit_author_role.role': sc.array(
                dims=['role'], values=list(roles.values())
            ),
        },
        schema=CORE_SCHEMA,
    )
