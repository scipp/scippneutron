# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import ClassVar, TypeAlias

import numpy as np
import scipp as sc

from . import _ir as ir

DataBlockName: TypeAlias = tuple[str, str]


class SqwFileType(enum.Enum):
    DND = 0
    SQW = 1


@dataclass(frozen=True, kw_only=True, slots=True)
class SqwFileHeader:
    prog_name: str
    prog_version: float
    sqw_type: SqwFileType
    n_dims: int


class SqwDataBlockType(enum.Enum):
    regular = "data_block"
    pix = "pix_data_block"
    dnd = "dnd_data_block"


@dataclass(frozen=True, kw_only=True, slots=True)
class SqwDataBlockDescriptor:
    block_type: SqwDataBlockType
    name: DataBlockName
    position: int  # u64
    size: int  # u32
    locked: bool  # u32


@dataclass(kw_only=True, slots=True)
class SqwMainHeader(ir.Serializable):
    full_filename: str
    title: str
    nfiles: int
    creation_date: datetime

    serial_name: ClassVar[str] = "main_header_cl"
    version: ClassVar[float] = 2.0

    def _serialize_to_dict(
        self,
    ) -> dict[str, ir.Object | ir.ObjectArray | ir.CellArray]:
        return {
            "serial_name": ir.String(self.serial_name),
            "version": ir.F64(self.version),
            "full_filename": ir.String(self.full_filename),
            "title": ir.String(self.title),
            "nfiles": ir.F64(float(self.nfiles)),
            "creation_date": ir.Datetime(self.creation_date),
            "creation_date_defined_privately": ir.Logical(False),
        }

    def prepare_for_serialization(self, filename: str, filepath: str) -> SqwMainHeader:
        return replace(self, creation_date=datetime.now(tz=timezone.utc))


@dataclass(kw_only=True, slots=True)
class SqwLineAxes(ir.Serializable):
    title: str
    label: list[str]
    img_scales: list[sc.Variable]
    img_range: list[sc.Variable]
    n_bins_all_dims: sc.Variable  # shape=(n_dim,) dtype=float64 [encodes int]
    single_bin_defines_iax: sc.Variable  # shape=(n_dim,) dtype=bool
    dax: sc.Variable
    offset: list[sc.Variable]
    changes_aspect_ratio: bool
    filename: str = ""
    filepath: str = ""

    serial_name: ClassVar[str] = "line_axes"
    version: ClassVar[float] = 7.0

    def _serialize_to_dict(
        self,
    ) -> dict[str, ir.Object | ir.ObjectArray | ir.CellArray]:
        units = ["1/angstrom"] * 3 + ["meV"]  # depends on SqwLineProj.type

        return {
            "serial_name": ir.String(self.serial_name),
            "version": ir.F64(self.version),
            "filename": ir.String(self.filename),
            "filepath": ir.String(self.filepath),
            "title": ir.String(self.title),
            "label": _serialize_str_array(self.label),
            "img_scales": _serialize_multi_unit_array(self.img_scales, units),
            "img_range": _serialize_multi_unit_array(self.img_range, units),
            "nbins_all_dims": _variable_to_float_array(self.n_bins_all_dims, None),
            "single_bin_defines_iax": ir.ObjectArray(
                shape=self.n_bins_all_dims.shape,
                data=[ir.Logical(bool(b)) for b in self.single_bin_defines_iax.values],
                ty=ir.TypeTag.logical,
            ),
            # +1 to convert to 1-based indexing
            "dax": _variable_to_float_array(self.dax + sc.index(1), None),
            "offset": _serialize_multi_unit_array(self.offset, units),
            "changes_aspect_ratio": ir.Logical(self.changes_aspect_ratio),
        }


@dataclass(kw_only=True, slots=True)
class SqwLineProj(ir.Serializable):
    lattice_spacing: sc.Variable  # vector
    lattice_angle: sc.Variable  # vector
    offset: list[sc.Variable]
    title: str
    label: list[str]
    u: sc.Variable  # vector
    v: sc.Variable  # vector
    w: sc.Variable | None  # vector
    non_orthogonal: bool
    type: str

    serial_name: ClassVar[str] = "line_proj"
    version: ClassVar[float] = 7.0

    def _serialize_to_dict(
        self,
    ) -> dict[str, ir.Object | ir.ObjectArray | ir.CellArray]:
        if self.type != "aaa":
            raise NotImplementedError(f"Projection type not supported: {self.type}")
        units = ["1/angstrom"] * 3 + ["meV"]  # depends on SqwLineProj.type

        if self.w is None:
            w = ir.Array(np.array([]), ir.TypeTag.f64)
        else:
            w = _variable_to_float_array(self.w, "1/angstrom")

        return {
            "serial_name": ir.String(self.serial_name),
            "version": ir.F64(self.version),
            "alatt": _variable_to_float_array(self.lattice_spacing, "angstrom"),
            "angdeg": _variable_to_float_array(self.lattice_angle, "deg"),
            "offset": _serialize_multi_unit_array(self.offset, units),
            "title": ir.String(self.title),
            "label": _serialize_str_array(self.label),
            "u": _variable_to_float_array(self.u, "1/angstrom"),
            "v": _variable_to_float_array(self.v, "1/angstrom"),
            "w": w,
            "nonorthogonal": ir.Logical(self.non_orthogonal),
            "type": ir.String(self.type),
        }


@dataclass(kw_only=True, slots=True)
class SqwDndMetadata(ir.Serializable):
    axes: SqwLineAxes
    proj: SqwLineProj
    creation_date: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    serial_name: ClassVar[str] = "dnd_metadata"
    version: ClassVar[float] = 1.0

    def _serialize_to_dict(
        self,
    ) -> dict[str, ir.Object | ir.ObjectArray | ir.CellArray]:
        axes = self.axes.serialize_to_ir()
        proj = self.proj.serialize_to_ir()

        return {
            "serial_name": ir.String(self.serial_name),
            "version": ir.F64(self.version),
            "axes": ir.ObjectArray(
                ty=ir.TypeTag.struct,
                shape=(1,),
                data=[axes],
            ),
            "proj": ir.ObjectArray(
                ty=ir.TypeTag.struct,
                shape=(1,),
                data=[proj],
            ),
            "creation_date_str": ir.Datetime(self.creation_date),
        }

    def prepare_for_serialization(self, filename: str, filepath: str) -> SqwDndMetadata:
        return replace(
            self,
            creation_date=datetime.now(tz=timezone.utc),
            axes=replace(self.axes, filename=filename, filepath=filepath),
        )


@dataclass(kw_only=True, slots=True)
class SqwPixelMetadata(ir.Serializable):
    full_filename: str
    npix: int
    data_range: np.ndarray[tuple[int, int], np.dtype[np.float64]]

    serial_name: ClassVar[str] = "pix_metadata"
    version: ClassVar[float] = 1.0

    def _serialize_to_dict(
        self,
    ) -> dict[str, ir.Object | ir.ObjectArray | ir.CellArray]:
        return {
            "serial_name": ir.String(self.serial_name),
            "version": ir.F64(self.version),
            "full_filename": ir.String(self.full_filename),
            "npix": ir.F64(float(self.npix)),
            "data_range": ir.Array(self.data_range, ty=ir.TypeTag.f64),
        }


@dataclass(kw_only=True, slots=True)
class SqwPixWrap(ir.Serializable):
    """Represents pixel data but does not hold the actual data."""

    n_rows: int = 9
    n_pixels: int

    def _serialize_to_dict(
        self,
    ) -> dict[str, ir.Object | ir.ObjectArray | ir.CellArray]:
        return {
            "n_rows": ir.U32(self.n_rows),
            "n_pixels": ir.U64(self.n_pixels),
        }


@dataclass(kw_only=True, slots=True)
class SqwIXSource(ir.Serializable):
    name: str
    target_name: str
    frequency: sc.Variable

    serial_name: ClassVar[str] = "IX_source"
    version: ClassVar[float] = 2.0

    def _serialize_to_dict(
        self,
    ) -> dict[str, ir.Object | ir.ObjectArray | ir.CellArray]:
        return {
            "serial_name": ir.String(self.serial_name),
            "version": ir.F64(self.version),
            "name": ir.String(self.name),
            "target_name": ir.String(self.target_name),
            "frequency": ir.F64(self.frequency.value),  # TODO unit
        }


@dataclass(kw_only=True, slots=True)
class SqwIXNullInstrument(ir.Serializable):
    name: str
    source: SqwIXSource

    serial_name: ClassVar[str] = "IX_null_inst"
    version: ClassVar[float] = 2.0

    def _serialize_to_dict(
        self,
    ) -> dict[str, ir.Object | ir.ObjectArray | ir.CellArray]:
        return {
            "serial_name": ir.String(self.serial_name),
            "version": ir.F64(self.version),
            "source": self.source.serialize_to_ir(),
            "name": ir.String(self.name),
        }


class EnergyMode(enum.Enum):
    direct = 1
    indirect = 2


@dataclass(kw_only=True, slots=True)
class SqwIXSample(ir.Serializable):
    name: str
    lattice_spacing: sc.Variable  # vector
    lattice_angle: sc.Variable  # vector

    serial_name: ClassVar[str] = "IX_sample"
    version: ClassVar[float] = 3.0

    def _serialize_to_dict(
        self,
    ) -> dict[str, ir.Object | ir.ObjectArray | ir.CellArray]:
        return {
            "serial_name": ir.String(self.serial_name),
            "version": ir.F64(self.version),
            "alatt": _variable_to_float_array(self.lattice_spacing, "angstrom"),
            "angdeg": _variable_to_float_array(self.lattice_angle, "deg"),
            "name": ir.String(self.name),
        }


# In contrast to SQW files, this model contains the nested
# struct fields instead of a nested struct in `array-dat`.
@dataclass(kw_only=True, slots=True)
class SqwIXExperiment(ir.Serializable):
    run_id: int
    # 1 element for direct, array of detector.shape for indirect
    efix: sc.Variable  # array or scalar
    emode: EnergyMode
    en: sc.Variable  # array
    psi: sc.Variable  # scalar
    u: sc.Variable  # vector
    v: sc.Variable  # vector
    omega: sc.Variable  # scalar
    dpsi: sc.Variable  # scalar
    gl: sc.Variable  # scalar
    gs: sc.Variable  # scalar
    filename: str = ""
    filepath: str = ""

    serial_name: ClassVar[str] = "IX_experiment"
    version: ClassVar[float] = 3.0

    def _serialize_to_dict(
        self,
    ) -> dict[str, ir.Object | ir.ObjectArray | ir.CellArray]:
        en = self.en.to(unit="meV", dtype="float64", copy=False)
        if en.ndim == 1:  # direct mode; still needs 2D en like indirect mode
            en = en.broadcast(sizes={"_": 1, "energy_transfer": self.en.shape[0]})
        else:
            en = en.transpose(dims=["detector", "energy_transfer"])

        efix = self.efix.to(unit="meV", dtype="float64", copy=False)
        if efix.ndim == 0:
            efix = efix.broadcast(sizes={"_": 1})
        return {
            "filename": ir.String(self.filename),
            "filepath": ir.String(self.filepath),
            "run_id": ir.F64(float(self.run_id + 1)),
            "efix": ir.Array(efix.values, ty=ir.TypeTag.f64),
            "emode": ir.F64(float(self.emode.value)),
            "en": ir.Array(en.values, ty=ir.TypeTag.f64),
            "psi": ir.F64(_angle_value(self.psi)),
            "u": ir.Array(self.u.values, ty=ir.TypeTag.f64),
            "v": ir.Array(self.v.values, ty=ir.TypeTag.f64),
            "omega": ir.F64(_angle_value(self.omega)),
            "dpsi": ir.F64(_angle_value(self.dpsi)),
            "gl": ir.F64(_angle_value(self.gl)),
            "gs": ir.F64(_angle_value(self.gs)),
            "angular_is_degree": ir.Logical(False),
            # serial_name and version are serialized by SqwMultiIXExperiment
        }


@dataclass(slots=True)
class SqwMultiIXExperiment(ir.Serializable):
    array_dat: list[SqwIXExperiment]

    serial_name: ClassVar[str] = "IX_experiment"
    version: ClassVar[float] = 3.0

    def _serialize_to_dict(
        self,
    ) -> dict[str, ir.Object | ir.ObjectArray | ir.CellArray]:
        return {
            "serial_name": ir.String(self.serial_name),
            "version": ir.F64(self.version),
            "array_dat": ir.ObjectArray(
                ty=ir.TypeTag.struct,
                shape=(len(self.array_dat),),
                data=[exp.serialize_to_ir() for exp in self.array_dat],
            ),
        }


@dataclass(kw_only=True, slots=True)
class UniqueRefContainer(ir.Serializable):
    global_name: str
    objects: UniqueObjContainer

    serial_name: ClassVar[str] = "unique_references_container"
    version: ClassVar[float] = 1.0

    def _serialize_to_dict(
        self,
    ) -> dict[str, ir.Object | ir.ObjectArray | ir.CellArray]:
        return {
            "serial_name": ir.String(self.serial_name),
            "version": ir.F64(self.version),
            "stored_baseclass": ir.String(self.objects.baseclass),
            "global_name": ir.String(self.global_name),
            "unique_objects": self.objects.serialize_to_ir().to_object_array(),
        }


@dataclass(kw_only=True, slots=True)
class UniqueObjContainer(ir.Serializable):
    baseclass: str
    objects: list[ir.Serializable]
    indices: list[int]

    serial_name: ClassVar[str] = "unique_objects_container"
    version: ClassVar[float] = 1.0

    def _serialize_to_dict(
        self,
    ) -> dict[str, ir.Object | ir.ObjectArray | ir.CellArray]:
        return {
            "serial_name": ir.String(self.serial_name),
            "version": ir.F64(self.version),
            "baseclass": ir.String(self.baseclass),
            "unique_objects": ir.CellArray(
                shape=(len(self.objects),),
                data=[obj.serialize_to_ir().to_object_array() for obj in self.objects],
            ),
            "idx": ir.ObjectArray(
                shape=(len(self.indices),),
                # +1 to convert to 1-based indexing
                data=np.array(self.indices) + 1.0,
                ty=ir.TypeTag.f64,
            ),
        }


def _angle_value(x: sc.Variable) -> float:
    return x.to(unit="rad", dtype="float64", copy=False).value  # type: ignore[no-any-return]


def _serialize_str_array(strings: list[str]) -> ir.CellArray:
    return ir.CellArray(
        shape=(len(strings),),
        data=[
            ir.ObjectArray(shape=(len(s),), data=[ir.String(s)], ty=ir.TypeTag.char)
            for s in strings
        ],
    )


def _serialize_multi_unit_array(data: list[sc.Variable], units: list[str]) -> ir.Array:
    stacked = np.stack(
        [d.to(unit=u, dtype="float64").values for d, u in zip(data, units, strict=True)]
    )
    return ir.Array(stacked, ty=ir.TypeTag.f64)


def _variable_to_float_array(var: sc.Variable, unit: str | None) -> ir.Array:
    if unit is not None:
        var = var.to(unit=unit, copy=False)
    elif var.unit is not None:
        raise sc.UnitError(f"Expected no unit, got: {var.unit}")
    return ir.Array(
        var.values.astype("float64", copy=False),
        ir.TypeTag.f64,
    )
