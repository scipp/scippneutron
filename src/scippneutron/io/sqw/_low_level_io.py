# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import functools
import struct
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, ParamSpec, TypeVar

import numpy as np
import numpy.typing as npt

from ._bytes import Byteorder

_P = ParamSpec("_P")
_R = TypeVar("_R")
_E = TypeVar("_E", bound=np.generic, covariant=True)


def _annotate_read_exception(
    ty: str,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Add a note with file-information to exceptions from read_* functions."""

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return func(*args, **kwargs)
            except (ValueError, UnicodeDecodeError) as exc:
                sqw_io: LowLevelSqw = args[0]  # type: ignore[assignment]
                _add_note_to_read_exception(exc, sqw_io, ty)
                raise

        return wrapper

    return decorator


def _add_note_to_read_exception(exc: Exception, sqw_io: LowLevelSqw, ty: str) -> None:
    path_piece = (
        "in-memory SQW file" if sqw_io.path is None else f"SQW file '{sqw_io.path}'"
    )
    _add_note(
        exc,
        f"When reading a {ty} from {path_piece} at position {sqw_io.position}",
    )


def _annotate_write_exception(
    ty: str,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Add a note with file-information to exceptions from write_* functions."""

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return func(*args, **kwargs)
            except (ValueError, UnicodeEncodeError, OverflowError) as exc:
                sqw_io: LowLevelSqw = args[0]  # type: ignore[assignment]
                _add_note_to_write_exception(exc, sqw_io, ty)
                raise

        return wrapper

    return decorator


def _add_note_to_write_exception(exc: Exception, sqw_io: LowLevelSqw, ty: str) -> None:
    path_piece = (
        "in-memory SQW file" if sqw_io.path is None else f"SQW file '{sqw_io.path}'"
    )
    _add_note(
        exc,
        f"When writing a {ty} to {path_piece} at position {sqw_io.position}",
    )


def _add_note(exc: Exception, note: str) -> None:
    try:
        exc.add_note(note)  # type: ignore[attr-defined]
    except AttributeError:
        # Python < 3.11 -> do nothing and accept throwing a worse error.
        pass


class LowLevelSqw:
    def __init__(
        self, file: BinaryIO, *, path: Path | None, byteorder: Byteorder | None = None
    ) -> None:
        self._file = file
        self._byteorder = _deduce_byteorder(self._file, byteorder=byteorder)
        self._path = path

    @_annotate_read_exception("logical")
    def read_logical(self) -> bool:
        buf = self._file.read(1)
        return buf != b"\x00"

    @_annotate_read_exception("u8")
    def read_u8(self) -> int:
        buf = self._file.read(1)
        return int.from_bytes(buf, self._byteorder.get())

    @_annotate_read_exception("u32")
    def read_u32(self) -> int:
        buf = self._file.read(4)
        return int.from_bytes(buf, self._byteorder.get())

    @_annotate_read_exception("u32")
    def read_u64(self) -> int:
        buf = self._file.read(8)
        return int.from_bytes(buf, self._byteorder.get())

    @_annotate_read_exception("f64")
    def read_f64(self) -> float:
        buf = self._file.read(8)
        match self._byteorder:
            case Byteorder.little:
                bo = "<"
            case Byteorder.big:
                bo = ">"
        return struct.unpack(bo + "d", buf)[0]  # type: ignore[no-any-return]

    @_annotate_read_exception("char array")
    def read_char_array(self) -> str:
        size = self.read_u32()
        return self.read_n_chars(size)

    @_annotate_read_exception("n chars")
    def read_n_chars(self, n: int) -> str:
        return self._file.read(n).decode("utf-8")

    @_annotate_read_exception("array")
    def read_array(
        self, shape: tuple[int, ...], dtype: np.dtype[_E]
    ) -> npt.NDArray[_E]:
        if not shape:
            return np.array([], dtype=dtype)

        count = int(np.prod(shape))
        dtype = dtype.newbyteorder(self.byteorder.value)
        if isinstance(self._file, BytesIO):
            flat = np.frombuffer(
                self._file.getbuffer(), offset=self.position, dtype=dtype, count=count
            )
            # Make a copy because np.frombuffer creates a view of the buffer.
            # This makes it impossible to write to the buffer while the view exists.
            # But when writing pixel data, we need to both
            # read and write from / to the same buffer.
            flat = flat.copy()
            self._file.seek(self.position + count * dtype.itemsize)
        else:
            flat = np.fromfile(self._file, dtype=dtype, count=int(np.prod(shape)))
        # Invert the shape because files use column-major layout.
        return flat.reshape(shape[::-1])

    @_annotate_write_exception("logical")
    def write_logical(self, value: bool) -> None:
        self._file.write(value.to_bytes(1, self._byteorder.get()))

    @_annotate_write_exception("u8")
    def write_u8(self, value: int) -> None:
        self._file.write(value.to_bytes(1, self._byteorder.get()))

    @_annotate_write_exception("u32")
    def write_u32(self, value: int) -> None:
        self._file.write(value.to_bytes(4, self._byteorder.get()))

    @_annotate_write_exception("u64")
    def write_u64(self, value: int) -> None:
        self._file.write(value.to_bytes(8, self._byteorder.get()))

    @_annotate_write_exception("f64")
    def write_f64(self, value: float) -> None:
        match self._byteorder:
            case Byteorder.little:
                bo = "<"
            case Byteorder.big:
                bo = ">"
        self._file.write(struct.pack(bo + "d", value))

    @_annotate_write_exception("char array")
    def write_char_array(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.write_u32(len(encoded))
        self._file.write(encoded)

    @_annotate_write_exception("char array")
    def write_chars(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self._file.write(encoded)

    @_annotate_write_exception("array")
    def write_array(
        self, array: npt.NDArray[np.float64] | npt.NDArray[np.float32]
    ) -> None:
        out = array.astype(array.dtype.newbyteorder(self.byteorder.value), copy=False)
        if isinstance(self._file, BytesIO):
            # Inefficient because it constructs an entire separate buffer in memory.
            # Could be optimised to write in chunks if need be.
            self._file.write(out.tobytes())
        else:
            out.tofile(self._file)

    @_annotate_write_exception("bytes")
    def write_raw(self, value: bytes | memoryview) -> None:
        self._file.write(value)

    def seek(self, pos: int) -> None:
        self._file.seek(pos)

    @property
    def byteorder(self) -> Byteorder:
        return self._byteorder

    @property
    def position(self) -> int:
        return self._file.tell()

    @property
    def path(self) -> Path | None:
        return self._path


def _deduce_byteorder(
    file: BinaryIO, *, byteorder: Byteorder | None = None
) -> Byteorder:
    """Guess the byte order of a file.

    The first four bytes of an SQW file are the length of the program name.
    Realistic lengths should be less than 2^16 bytes which is the flip over point
    between little and big endian.
    So we simply use the smaller number.

    This could be made more robust by reading (parts of) the rest of the header
    and checking that it makes sense, e.g., that the program name is valid UTF-8.
    But since HORACE ignores the issue of byteorder, what we have here should be enough.
    """
    if byteorder is not None:
        return byteorder

    pos = file.tell()
    buf = file.read(4)
    file.seek(pos)

    le_size = int.from_bytes(buf, "little")
    be_size = int.from_bytes(buf, "big")
    if le_size < be_size:
        return Byteorder.little
    return Byteorder.big
