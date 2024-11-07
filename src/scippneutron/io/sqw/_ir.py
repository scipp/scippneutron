# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""Intermediate representation for SQW objects."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, TypeVar

import numpy.typing as npt

_T = TypeVar("_T")


class TypeTag(enum.Enum):
    """Single byte tag to identify types in SQW files."""

    # Gaps in values are unsupported types.
    logical = 0
    char = 1
    f64 = 3
    f32 = 4
    i8 = 5
    u8 = 6
    i32 = 9
    u32 = 10
    i64 = 11
    u64 = 12
    cell = 23
    struct = 24
    serializable = 32  # objects that 'serialize themselves'


@dataclass(kw_only=True)
class ObjectArray:
    shape: tuple[int, ...]
    data: list[Object] | npt.NDArray[Any]
    ty: TypeTag


@dataclass(kw_only=True)
class CellArray:
    shape: tuple[int, ...]
    # nested object array to encode types of each item
    data: list[ObjectArray | CellArray]
    ty: ClassVar[TypeTag] = TypeTag.cell


@dataclass(kw_only=True)
class Struct:
    field_names: tuple[str, ...]
    field_values: CellArray
    ty: ClassVar[TypeTag] = TypeTag.struct

    def to_object_array(self) -> ObjectArray:
        return ObjectArray(
            ty=self.ty,
            shape=(1,),
            data=[self],
        )


@dataclass()
class String:
    value: str
    ty: ClassVar[TypeTag] = TypeTag.char


@dataclass()
class F64:
    value: float
    ty: ClassVar[TypeTag] = TypeTag.f64


@dataclass()
class U64:
    value: int
    ty: ClassVar[TypeTag] = TypeTag.u64


@dataclass()
class U32:
    value: int
    ty: ClassVar[TypeTag] = TypeTag.u32


@dataclass()
class U8:
    value: int
    ty: ClassVar[TypeTag] = TypeTag.u8


@dataclass()
class Logical:
    value: bool
    ty: ClassVar[TypeTag] = TypeTag.logical


# Not a dedicated type in SQW. Used here to encode NumPy arrays where list[ir.Object]
# is not efficient enough.
@dataclass()
class Array:
    value: npt.NDArray[Any]
    ty: TypeTag


# Not supported by SQW but represented here to simplify serialization.
@dataclass()
class Datetime:
    value: datetime
    ty: ClassVar[TypeTag] = TypeTag.char


Object = Struct | String | F64 | U64 | U32 | U8 | Logical | Array | Datetime


class Serializable(ABC):
    @abstractmethod
    def _serialize_to_dict(self) -> dict[str, Object | ObjectArray | CellArray]: ...

    def serialize_to_ir(self) -> Struct:
        fields = self._serialize_to_dict()
        return Struct(
            field_names=tuple(fields),
            field_values=CellArray(
                shape=(len(fields), 1),  # HORACE uses a 2D array
                data=[_serialize_field(field) for field in fields.values()],
            ),
        )

    def prepare_for_serialization(self: _T, filename: str, filepath: str) -> _T:  # noqa: PYI019
        return self


def _serialize_field(
    field: Object | ObjectArray | CellArray,
) -> ObjectArray | CellArray:
    if isinstance(field, ObjectArray | CellArray):
        return field
    if isinstance(field, Datetime):
        field = String(value=field.value.isoformat(timespec="seconds"))
    if isinstance(field, String):
        # TODO do we need to set the shape to empty?
        #  do we need to treat missing strings differently from empty strings?
        return ObjectArray(
            ty=field.ty, shape=(len(field.value),) if field.value else (), data=[field]
        )
    if isinstance(field, Array):
        return ObjectArray(ty=field.ty, shape=field.value.shape[::-1], data=field.value)
    return ObjectArray(ty=field.ty, shape=(1,), data=[field])
