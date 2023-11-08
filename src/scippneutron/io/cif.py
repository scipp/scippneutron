# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import io
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Union

import scipp as sc

from .table import serialize_to_table


def save_cif(
    fname: Union[str, Path, io.TextIOBase], blocks: Mapping[str, Mapping[str, Any]]
) -> None:
    with _open(fname) as f:
        for name, block in blocks.items():
            _save_block(f, name, block)


@contextmanager
def _open(fname: Union[str, Path, io.TextIOBase]):
    if isinstance(fname, io.TextIOBase):
        yield fname
    else:
        with open(fname, 'w') as f:
            yield f


def _save_block(f, name: str, block: Mapping[str, Any]) -> None:
    f.write(f'data_{name}\n')
    for key, val in block.items():
        _write_item(f, key, val)


def _write_loop(f, name: str, da: Union[sc.Dataset, sc.DataArray]) -> None:
    f.write('\nloop_\n')
    rows = serialize_to_table(da, units=False)
    header = next(rows)
    for label in header:
        f.write(f'_{name}.{label}\n')
    for row in rows:
        f.write(' '.join(row))
        f.write('\n')


def _format_value(value: Any) -> str:
    if isinstance(value, sc.Variable):
        without_unit = sc.scalar(value.value, variance=value.variance)
        return f'{without_unit:c}'
    return str(value)


def _write_key_value_pairs(f, name: str, pairs: Mapping[str, Any]) -> None:
    f.write('\n')
    for key, val in pairs.items():
        f.write(f'_{name}.{key} {_format_value(val)}\n')


def _write_item(f, name: str, value: Any) -> None:
    if isinstance(value, (sc.Dataset, sc.DataArray)):
        _write_loop(f, name, value)
    else:
        _write_key_value_pairs(f, name, value)
