# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Common file handling."""

from contextlib import AbstractContextManager, nullcontext
from io import BytesIO, StringIO
from os import PathLike
from typing import IO, Any, BinaryIO, Literal, TextIO, overload


@overload
def open_or_pass(
    path: StringIO,
    mode: Literal["r", "w", "r+"],
) -> AbstractContextManager[StringIO]: ...


@overload
def open_or_pass(
    path: str | PathLike[str] | TextIO,
    mode: Literal["r", "w", "r+"],
) -> AbstractContextManager[TextIO]: ...


@overload
def open_or_pass(
    path: BytesIO,
    mode: Literal["rb", "wb", "r+b"],
) -> AbstractContextManager[BytesIO]: ...


@overload
def open_or_pass(
    path: str | PathLike[str] | BinaryIO | BytesIO,
    mode: Literal["rb", "wb", "r+b"],
) -> AbstractContextManager[BinaryIO]: ...


def open_or_pass(
    path: str | PathLike[str] | TextIO | StringIO | BinaryIO | BytesIO,
    mode: Literal["r", "w", "r+", "rb", "wb", "r+b"],
) -> AbstractContextManager[IO[Any]]:
    """Open a file at a path or return an already open file."""
    if isinstance(path, TextIO | StringIO | BytesIO | BinaryIO):
        return nullcontext(path)
    return open(path, mode)
