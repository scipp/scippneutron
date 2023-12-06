# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import io
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

import scipp as sc


def save_cif(
    fname: Union[str, Path, io.TextIOBase], blocks: Union[Block, Iterable[Block]]
) -> None:
    if isinstance(blocks, Block):
        blocks = (blocks,)
    with _open(fname) as f:
        _write_file_heading(f)
        _write_multi(f, blocks)


class Chunk:
    def __init__(
        self,
        pairs: Union[Mapping[str, Any], Iterable[tuple[str, Any]], None],
        /,
        *,
        comment: str = '',
    ) -> None:
        self._pairs = dict(pairs) if pairs is not None else {}
        self._comment = _encode_non_ascii(comment)

    @property
    def comment(self) -> str:
        return self._comment

    @comment.setter
    def comment(self, comment: str) -> None:
        self._comment = _encode_non_ascii(comment)

    def write(self, f: io.TextIOBase) -> None:
        _write_comment(f, self.comment)
        for key, val in self._pairs.items():
            v = _format_value(val)
            if v.startswith(';'):
                f.write(f'_{key}\n{v}\n')
            else:
                f.write(f'_{key} {v}\n')


class Loop:
    def __init__(
        self,
        columns: Union[
            Mapping[str, sc.Variable], Iterable[tuple[str, sc.Variable]], None
        ],
        *,
        comment: str = '',
    ) -> None:
        self._columns = dict(columns) if columns is not None else {}
        self._comment = _encode_non_ascii(comment)

    @property
    def comment(self) -> str:
        return self._comment

    @comment.setter
    def comment(self, comment: str) -> None:
        self._comment = _encode_non_ascii(comment)

    def write(self, f: io.TextIOBase) -> None:
        _write_comment(f, self.comment)
        f.write('loop_\n')
        for key in self._columns:
            f.write(f'_{key}\n')
        formatted_values = [
            tuple(map(_format_value, row)) for row in zip(*self._columns.values())
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
    def __init__(
        self,
        name: str,
        content: Optional[Iterable[Union[Mapping[str, Any], Loop, Chunk]]] = None,
        *,
        comment: str = '',
    ) -> None:
        self._name = _encode_non_ascii(name)
        self._content = _convert_input_content(content) if content is not None else []
        self._comment = _encode_non_ascii(comment)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = _encode_non_ascii(name)

    @property
    def comment(self) -> str:
        return self._comment

    @comment.setter
    def comment(self, comment: str) -> None:
        self._comment = _encode_non_ascii(comment)

    def add(
        self,
        content: Union[Mapping[str, Any], Iterable[tuple[str, Any]], Chunk],
        /,
        comment: str = '',
    ) -> None:
        if not isinstance(content, Chunk):
            content = Chunk(content, comment=comment)
        self._content.append(content)

    def write(self, f: io.TextIOBase) -> None:
        _write_comment(f, self.comment)
        f.write(f'data_{self.name}\n\n')
        _write_multi(f, self._content)

    def save(
        self,
        fname: Union[str, Path, io.TextIOBase],
    ) -> None:
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
