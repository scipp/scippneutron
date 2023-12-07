# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""CIF file writer.

This module contains tools for writing `CIF <https://www.iucr.org/resources/cif>`_
files with diffraction data.
It does not support reading CIF files.

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

Wrap the data in a ``Loop`` to write them together as columns:

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

import io
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

import scipp as sc


@dataclass(frozen=True)
class CIFSchema:
    name: str
    version: str
    location: str


CORE_SCHEMA = CIFSchema(
    name='coreCIF',
    version='3.3.0',
    location='https://github.com/COMCIFS/cif_core/blob/fc3d75a298fd7c0c3cde43633f2a8616e826bfd5/cif_core.dic',  # noqa: E501
)
PD_SCHEMA = CIFSchema(
    name='pdCIF',
    version='2.5.0',
    location='https://github.com/COMCIFS/Powder_Dictionary/blob/7608b92165f58f968f054344e67662e01d4b401a/cif_pow.dic',  # noqa: E501
)


def save_cif(
    fname: Union[str, Path, io.TextIOBase], blocks: Union[Block, Iterable[Block]]
) -> None:
    """Save data blocks to a CIF file.

    To use, first create :class:`scippneutron.io.cif.Block` objects to collect and
    structure data for the file, then use this function to write the file.

    Parameters
    ----------
    fname:
        Path or file handle for the output file.
    blocks:
        One or more CIF data blocks to write to the file.

    See also
    --------
    scippneutron.io.cif.Block.save:
        Method for saving a single block.
    """
    if isinstance(blocks, Block):
        blocks = (blocks,)
    with _open(fname) as f:
        _write_file_heading(f)
        _write_multi(f, blocks)


class Chunk:
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
        pairs: Union[Mapping[str, Any], Iterable[tuple[str, Any]], None],
        /,
        *,
        comment: str = '',
        schema: Optional[Union[CIFSchema, Iterable[CIFSchema]]] = None,
    ) -> None:
        """Create a new CIF chunk.

        Parameters
        ----------
        pairs:
            Defines a mapping from keys (a.k.a. tags) to values.
        comment:
            Optional comment that can be written above the chunk in the file.
        schema:
            CIF Schema used for the chunk.
            Content is not checked against the schema, but the schema is written
            to the file.
        """
        self._pairs = dict(pairs) if pairs is not None else {}
        self._comment = _encode_non_ascii(comment)
        self._schema = _preprocess_schema(schema)

    @property
    def comment(self) -> str:
        """Optional comment that can be written above the chunk in the file."""
        return self._comment

    @comment.setter
    def comment(self, comment: str) -> None:
        self._comment = _encode_non_ascii(comment)

    @property
    def schema(self) -> set[CIFSchema]:
        """CIF Schema used for the chunk."""
        return self._schema

    def write(self, f: io.TextIOBase) -> None:
        """Write this chunk to a file.

        Used mainly internally, use :func:`scippneutron.io.cif.save_cif`
        or :meth:`scippneutron.io.cif.Block.save` instead.

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


class Loop:
    """A CIF loop.

    Contains a mapping from strings to Scipp variables.
    The strings are arbitrary and ``Loop`` can merge items from different categories
    into a single loop.
    All variables must have the same length.
    """

    def __init__(
        self,
        columns: Union[
            Mapping[str, sc.Variable], Iterable[tuple[str, sc.Variable]], None
        ],
        *,
        comment: str = '',
        schema: Optional[Union[CIFSchema, Iterable[CIFSchema]]] = None,
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
            CIF Schema used for the loop.
            Content is not checked against the schema, but the schema is written
            to the file.
        """
        self._columns = dict(columns) if columns is not None else {}
        self._comment = _encode_non_ascii(comment)
        self._schema = _preprocess_schema(schema)

    @property
    def comment(self) -> str:
        """Optional comment that can be written above the loop in the file."""
        return self._comment

    @comment.setter
    def comment(self, comment: str) -> None:
        self._comment = _encode_non_ascii(comment)

    @property
    def schema(self) -> set[CIFSchema]:
        """CIF Schema used for the loop."""
        return self._schema

    def write(self, f: io.TextIOBase) -> None:
        """Write this loop to a file.

        Used mainly internally, use :func:`scippneutron.io.cif.save_cif`
        or :meth:`scippneutron.io.cif.Block.save` instead.

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
            for row in _strict_zip(*self._columns.values())
        ]
        sep = (
            '\n'
            if any(';' in item for row in formatted_values for item in row)
            else ' '
        )
        for row in formatted_values:
            f.write(sep.join(row))
            f.write('\n')


class Block:
    """A CIF data block.

    A block contains an ordered sequence of loops
    and chunks (groups of key-value-pairs).
    The contents are written to file in the order specified in the block.
    """

    def __init__(
        self,
        name: str,
        content: Optional[Iterable[Union[Mapping[str, Any], Loop, Chunk]]] = None,
        *,
        comment: str = '',
        schema: Optional[Union[CIFSchema, Iterable[CIFSchema]]] = None,
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
            CIF Schema used for the block.
            Content is not checked against the schema, but the schema is written
            to the file.
        """
        self._name = ''
        self.name = name
        self._content = _convert_input_content(content) if content is not None else []
        self._comment = _encode_non_ascii(comment)
        self._schema = _preprocess_schema(schema)

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
    def comment(self) -> str:
        """Optional comment that can be written above the block in the file."""
        return self._comment

    @comment.setter
    def comment(self, comment: str) -> None:
        self._comment = _encode_non_ascii(comment)

    @property
    def schema(self) -> set[CIFSchema]:
        """CIF Schema used for the block."""
        merged = set(self._schema)
        for item in self._content:
            merged.update(item.schema)
        return merged

    def add(
        self,
        content: Union[Mapping[str, Any], Iterable[tuple[str, Any]], Chunk],
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
        if not isinstance(content, Chunk):
            content = Chunk(content, comment=comment)
        self._content.append(content)

    def write(self, f: io.TextIOBase) -> None:
        """Write this block to a file.

        Used mainly internally, use :func:`scippneutron.io.cif.save_cif`
        or :meth:`scippneutron.io.cif.Block.save` instead.

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

    def save(
        self,
        fname: Union[str, Path, io.TextIOBase],
    ) -> None:
        """Save this block to a CIF file.

        Equivalent to ``cif.save_cif(fname, self)``.

        Parameters
        ----------
        fname:
            Path or file handle for the output file.

        See also
        --------
        scippneutron.io.cif.save_cif:
            Free function for saving one or more blocks.
        """
        save_cif(fname, self)


def _convert_input_content(
    content: Iterable[Union[Mapping[str, Any], Loop, Chunk]]
) -> list[Union[Loop, Chunk]]:
    return [
        item if isinstance(item, (Loop, Chunk)) else Chunk(item) for item in content
    ]


@contextmanager
def _open(fname: Union[str, Path, io.TextIOBase]):
    if isinstance(fname, io.TextIOBase):
        yield fname
    else:
        with open(fname, 'w') as f:
            yield f


def _preprocess_schema(
    schema: Optional[Union[CIFSchema, Iterable[CIFSchema]]]
) -> set[CIFSchema]:
    if schema is None:
        return set()
    if isinstance(schema, CIFSchema):
        res = {schema}
    else:
        res = set(schema)
    res.add(CORE_SCHEMA)  # needed to encode schema itself
    return res


def _make_schema_loop(schema: set[CIFSchema]) -> Optional[Loop]:
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


def _quotes_for_string_value(value: str) -> Optional[str]:
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


def _write_comment(f: io.TextIOBase, comment: str) -> None:
    if comment:
        f.write('# ')
        f.write('\n# '.join(comment.splitlines()))
        f.write('\n')


def _write_multi(f: io.TextIOBase, to_write: Iterable[Any]) -> None:
    first = True
    for item in to_write:
        if not first:
            f.write('\n')
        first = False
        item.write(f)


def _write_file_heading(f: io.TextIOBase) -> None:
    f.write('#\\#CIF_1.1\n')


def _strict_zip(*args: Iterable[Any]) -> Iterable[Any]:
    try:
        return zip(*args, strict=True)
    except TypeError:
        pass
    return zip(*args)
