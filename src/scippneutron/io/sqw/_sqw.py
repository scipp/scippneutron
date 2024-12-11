# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import warnings
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np
import numpy.typing as npt
import scipp as sc
from dateutil.parser import parse as parse_datetime

from .._files import open_or_pass
from . import _ir as ir
from ._build import SqwBuilder
from ._bytes import Byteorder
from ._low_level_io import LowLevelSqw
from ._models import (
    DataBlockName,
    EnergyMode,
    SqwDataBlockDescriptor,
    SqwDataBlockType,
    SqwDndMetadata,
    SqwFileHeader,
    SqwFileType,
    SqwIXExperiment,
    SqwIXNullInstrument,
    SqwIXSample,
    SqwIXSource,
    SqwLineAxes,
    SqwLineProj,
    SqwMainHeader,
    SqwPixelMetadata,
)
from ._read_write import read_object_array


class Sqw:
    def __init__(
        self,
        *,
        sqw_io: LowLevelSqw,
        file_header: SqwFileHeader,
        block_allocation_table: dict[DataBlockName, SqwDataBlockDescriptor],
    ) -> None:
        self._sqw_io = sqw_io
        self._file_header = file_header
        self._block_allocation_table = block_allocation_table

    @classmethod
    @contextmanager
    def open(
        cls,
        path: str | PathLike[str] | BinaryIO | BytesIO,
        *,
        byteorder: Byteorder | Literal["little", "big"] | None = None,
    ) -> Generator[Sqw, None, None]:
        with open_or_pass(path, "rb") as f:
            stored_path = None if isinstance(path, BinaryIO | BytesIO) else Path(path)
            sqw_io = LowLevelSqw(
                f,
                path=stored_path,
                byteorder=Byteorder.parse(byteorder) if byteorder is not None else None,
            )
            file_header = _read_file_header(sqw_io)
            data_block_descriptors = _read_block_allocation_table(sqw_io)
            yield Sqw(
                sqw_io=sqw_io,
                file_header=file_header,
                block_allocation_table=data_block_descriptors,
            )

    @classmethod
    def build(
        cls,
        path: str | PathLike[str] | BinaryIO | BytesIO,
        *,
        title: str = "",
        byteorder: Byteorder | Literal["native", "little", "big"] = "native",
    ) -> SqwBuilder:
        return SqwBuilder(path, title, byteorder=Byteorder.parse(byteorder))

    @property
    def file_header(self) -> SqwFileHeader:
        return self._file_header

    @property
    def byteorder(self) -> Byteorder:
        return self._sqw_io.byteorder

    def data_block_names(self) -> Iterable[DataBlockName]:
        """Return an iterator over the names of all stored data blocks."""
        return self._block_allocation_table.keys()

    # TODO lock blocks during writing, esp pix data
    def read_data_block(
        self, name: DataBlockName | str, level2_name: str | None = None, /
    ) -> Any:  # TODO type
        block_name = _normalize_data_block_name(name, level2_name)
        try:
            block_descriptor = self._block_allocation_table[block_name]
        except KeyError:
            raise KeyError(f"No data block {block_name!r} in file") from None

        self._sqw_io.seek(block_descriptor.position)
        match block_descriptor.block_type:
            case SqwDataBlockType.regular:
                return _parse_block(read_object_array(self._sqw_io))
            case SqwDataBlockType.pix:
                return _read_pix_block(self._sqw_io)
            case SqwDataBlockType.dnd:
                return _read_dnd_block(self._sqw_io)
            case _:
                raise NotImplementedError(
                    f"Unsupported data block type: {block_descriptor.block_type}"
                )

    def __str__(self) -> str:
        path = self._sqw_io.path
        path_piece = "In-memory SQW file" if path is None else f"SQW file '{path}'"
        program = self._file_header.prog_name
        version = self._file_header.prog_version
        sqw_type = self._file_header.sqw_type
        n_dims = self._file_header.n_dims
        return f"{path_piece} ({sqw_type}, {program=}, {version=}, {n_dims=})"


def _read_file_header(sqw_io: LowLevelSqw) -> SqwFileHeader:
    prog_name = sqw_io.read_char_array()
    prog_version = sqw_io.read_f64()
    if prog_name != "horace" or prog_version != 4.0:
        warnings.warn(
            f"SQW program not supported: '{prog_name}' version {prog_version} "
            f"(expected 'horace' with version 4.0)",
            UserWarning,
            stacklevel=2,
        )

    sqw_type = SqwFileType(sqw_io.read_u32())
    if sqw_type != SqwFileType.SQW:
        warnings.warn("DND files are not supported", UserWarning, stacklevel=2)

    n_dims = sqw_io.read_u32()
    return SqwFileHeader(
        prog_name=prog_name, prog_version=prog_version, sqw_type=sqw_type, n_dims=n_dims
    )


def _read_block_allocation_table(
    sqw_io: LowLevelSqw,
) -> dict[DataBlockName, SqwDataBlockDescriptor]:
    _bat_size = sqw_io.read_u32()  # don't need this
    n_blocks = sqw_io.read_u32()
    descriptors = [_read_data_block_descriptor(sqw_io) for _ in range(n_blocks)]
    return {descriptor.name: descriptor for descriptor in descriptors}


def _read_data_block_descriptor(sqw_io: LowLevelSqw) -> SqwDataBlockDescriptor:
    block_type = SqwDataBlockType(sqw_io.read_char_array())
    name = sqw_io.read_char_array(), sqw_io.read_char_array()
    position = sqw_io.read_u64()
    size = sqw_io.read_u32()
    locked = sqw_io.read_u32() == 1
    return SqwDataBlockDescriptor(
        block_type=block_type, name=name, position=position, size=size, locked=locked
    )


def _normalize_data_block_name(
    name: DataBlockName | str, level2_name: str | None
) -> DataBlockName:
    match (name, level2_name):
        case (str(n1), str(n2)), None:
            return DataBlockName((n1, n2))  # type: ignore[no-any-return, operator]
        case str(n1), str(n2):
            return DataBlockName((n1, n2))  # type: ignore[no-any-return, operator]
    raise TypeError(
        "Data block name must be given either as a tuple of two strings or two "
        f"separate strings. Got {name!r} and {level2_name!r}."
    )


class AbortParse(Exception): ...


def _parse_block(block: ir.ObjectArray | ir.CellArray) -> Any:
    try:
        if isinstance(block, ir.CellArray):
            raise AbortParse("Block is a cell array, not a struct")
        return _try_parse_block(block)
    except AbortParse as abort:
        warnings.warn(
            f"Unable to parse SQW block: {abort.args[0]}", UserWarning, stacklevel=2
        )
        return block


def _try_parse_block(block: ir.ObjectArray) -> Any:
    if block.shape != (1,):
        raise AbortParse(f"Unsupported block shape: {block.shape}")
    if block.ty != ir.TypeTag.struct:
        raise AbortParse(f"Unsupported block type: {block.shape}, expected struct")
    struct: ir.Struct = block.data[0]  # type: ignore[assignment]
    if sum(s != 1 for s in struct.field_values.shape) != 1:
        raise AbortParse(
            "Contents cannot be a multi-dimensional cell array, "
            f"got shape {struct.field_values.shape}"
        )

    key = _get_struct_type_id(struct)
    try:
        parser = _BLOCK_PARSERS[key]
    except KeyError:
        raise AbortParse(f"No parser for struct type {key}") from None
    return parser(struct)


def _get_struct_type_id(struct: ir.Struct) -> tuple[str, float]:
    name = _get_struct_field(struct, "serial_name")
    if len(name.shape) != 1:
        raise AbortParse("'serial_name' is multi-dimensional")
    version = _get_struct_field(struct, "version")
    if version.shape != (1,):
        raise AbortParse("'version' is multi-dimensional")

    n = name.data[0]
    v = version.data[0]
    if not isinstance(n, ir.String):
        raise AbortParse("'serial_name' is not a string")
    if not isinstance(v, ir.F64):
        raise AbortParse("'version' is not an f64")

    return n.value, v.value


def _get_struct_field(struct: ir.Struct, name: str) -> ir.ObjectArray | ir.CellArray:
    for field_name, value in zip(
        struct.field_names, struct.field_values.data, strict=True
    ):
        if field_name == name:
            return value
    raise AbortParse(f"No field '{name}' in struct")


def _get_scalar_struct_field(struct: ir.Struct, name: str) -> Any:
    field = _get_struct_field(struct, name)
    shape = field.shape[1:] if field.ty == ir.TypeTag.char else field.shape
    if shape not in ((1,), ()):
        raise AbortParse(f"Field '{name}' has non-scalar shape: {shape}")
    if isinstance(field.data[0], ir.Struct | ir.ObjectArray | ir.CellArray):
        raise AbortParse(f"Field '{name}' contains a nested struct or cell array")
    return field.data[0].value


def _unpack_cell_array(cell_array: ir.CellArray) -> list[Any]:
    # This does not support general cell arrays.
    # It is specialized for 1d cell arrays of strings.
    data = (obj.data for obj in cell_array.data)
    return [d[0].value if len(d) == 1 else [x.value for x in d] for d in data]  # type: ignore[union-attr]


def _parse_main_header_cl_2_0(struct: ir.Struct) -> SqwMainHeader:
    return SqwMainHeader(
        full_filename=_get_scalar_struct_field(struct, "full_filename"),
        title=_get_scalar_struct_field(struct, "title"),
        nfiles=int(_get_scalar_struct_field(struct, "nfiles")),
        creation_date=parse_datetime(_get_scalar_struct_field(struct, "creation_date")),
    )


def _parse_dnd_metadata_1_0(struct: ir.Struct) -> SqwDndMetadata:
    (axes_struct,) = _get_struct_field(struct, "axes").data
    (proj_struct,) = _get_struct_field(struct, "proj").data
    proj, units = _parse_line_proj_7_0(proj_struct)  # type: ignore[arg-type]
    axes = _parse_line_axes_7_0(axes_struct, units)  # type: ignore[arg-type]
    return SqwDndMetadata(
        axes=axes,
        proj=proj,
        creation_date=parse_datetime(
            _get_scalar_struct_field(struct, "creation_date_str")
        ),
    )


def _parse_line_axes_7_0(struct: ir.Struct, units: list[str]) -> SqwLineAxes:
    return SqwLineAxes(
        title=_get_scalar_struct_field(struct, "title"),
        label=_unpack_cell_array(_get_struct_field(struct, "label")),
        img_scales=_parse_1d_multi_unit_array(
            _get_struct_field(struct, "img_scales").data, units
        ),
        img_range=_parse_2d_multi_unit_array(
            "range", _get_struct_field(struct, "img_range").data, units
        ),
        n_bins_all_dims=sc.array(
            dims=("axis",),
            values=_get_struct_field(struct, "nbins_all_dims").data,
            dtype="int64",
            unit=None,
        ),
        single_bin_defines_iax=sc.array(
            dims=("axis",),
            values=[
                d.value
                for d in _get_struct_field(struct, "single_bin_defines_iax").data
            ],
        ),
        dax=sc.array(
            dims=("axis",),
            # -1 to convert to 0-based indexing
            values=_get_struct_field(struct, "dax").data.astype(int) - 1,
            unit=None,
        ),
        offset=_parse_1d_multi_unit_array(
            _get_struct_field(struct, "offset").data, units
        ),
        changes_aspect_ratio=_get_scalar_struct_field(struct, "changes_aspect_ratio"),
        filename=_get_scalar_struct_field(struct, "filename"),
        filepath=_get_scalar_struct_field(struct, "filepath"),
    )


def _parse_line_proj_7_0(struct: ir.Struct) -> tuple[SqwLineProj, list[str]]:
    def get_vec(name: str, unit: str) -> sc.Variable | None:
        values = _get_struct_field(struct, name).data
        if values.shape == (0,):
            return None
        return sc.vector(values, unit=unit)

    ty = _get_scalar_struct_field(struct, "type")
    if ty != "aaa":
        # See https://pace-neutrons.github.io/Horace/v4.0.0/manual/Cutting_data_of_interest_from_SQW_files_and_objects.html#projection-in-details
        # for other units
        raise AbortParse(f"Unsupported 'type' in line_proj: {ty}")
    units = ["1/angstrom"] * 3 + ["meV"]

    return SqwLineProj(
        lattice_spacing=sc.vector(
            _get_struct_field(struct, "alatt").data, unit="1/angstrom"
        ),
        lattice_angle=sc.vector(_get_struct_field(struct, "angdeg").data, unit="deg"),
        offset=_parse_1d_multi_unit_array(
            _get_struct_field(struct, "offset").data,
            units,
        ),
        title=_get_scalar_struct_field(struct, "title"),
        label=_unpack_cell_array(_get_struct_field(struct, "label")),
        u=get_vec("u", unit=units[0]),
        v=get_vec("v", unit=units[1]),
        w=get_vec("w", unit=units[2]),
        non_orthogonal=_get_scalar_struct_field(struct, "nonorthogonal"),
        type=ty,
    ), units


def _parse_1d_multi_unit_array(
    values: npt.NDArray[Any], units: list[str]
) -> list[sc.Variable]:
    return [
        sc.scalar(value, unit=unit) for value, unit in zip(values, units, strict=False)
    ]


def _parse_2d_multi_unit_array(
    dim: str, values: npt.NDArray[Any], units: list[str]
) -> list[sc.Variable]:
    return [
        sc.array(dims=[dim], values=values, unit=unit)
        for values, unit in zip(values, units, strict=False)
    ]


def _parse_pix_metadata_1_0(struct: ir.Struct) -> SqwPixelMetadata:
    data_range = _get_struct_field(struct, "data_range").data
    if not isinstance(data_range, np.ndarray):
        raise AbortParse("'data_range' is not a numpy array")
    return SqwPixelMetadata(
        full_filename=_get_scalar_struct_field(struct, "full_filename"),
        npix=int(_get_scalar_struct_field(struct, "npix")),
        data_range=data_range,
    )


def _parse_ix_source_2_0(struct: ir.Struct) -> SqwIXSource:
    name = _get_scalar_struct_field(struct, "name")
    target_name = _get_scalar_struct_field(struct, "target_name")
    frequency = _get_scalar_struct_field(struct, "frequency")
    return SqwIXSource(
        name=name,
        target_name=target_name,
        # TODO unit currently unknown
        frequency=sc.scalar(frequency, unit=None),
    )


def _parse_ix_null_instrument_1_0(struct: ir.Struct) -> SqwIXNullInstrument:
    source = _try_parse_block(_get_struct_field(struct, "source"))
    name = _get_scalar_struct_field(struct, "name")
    return SqwIXNullInstrument(name=name, source=source)


def _parse_ix_sample_0_0(struct: ir.Struct) -> SqwIXSample:
    name = _get_scalar_struct_field(struct, "name")
    lattice_spacing = sc.vector(
        _get_struct_field(struct, "alatt").data, unit="1/angstrom"
    )
    lattice_angle = sc.vector(_get_struct_field(struct, "angdeg").data, unit="deg")
    return SqwIXSample(
        name=name,
        lattice_spacing=lattice_spacing,
        lattice_angle=lattice_angle,
    )


def _parse_ix_experiment_3_0(struct: ir.Struct) -> list[SqwIXExperiment]:
    return [
        _parse_single_ix_experiment_3_0(run)
        for run in _get_struct_field(struct, "array_dat").data
    ]


def _parse_single_ix_experiment_3_0(struct: ir.Struct) -> SqwIXExperiment:
    def g(n: str) -> Any:
        return _get_scalar_struct_field(struct, n)

    candidate_efix = _get_struct_field(struct, "efix").data
    if isinstance(candidate_efix, np.ndarray):
        efix = sc.array(dims=["detector"], values=candidate_efix, unit="meV")
    else:
        (e,) = candidate_efix
        efix = sc.scalar(e.value, unit="meV")

    raw_en = _get_struct_field(struct, "en").data
    if isinstance(raw_en, np.ndarray):
        en = raw_en.squeeze()
    else:
        en = [e.value for e in raw_en]

    angle_unit = sc.Unit("deg" if g("angular_is_degree") else "rad")

    return SqwIXExperiment(
        filename=g("filename"),
        filepath=g("filepath"),
        run_id=int(g("run_id")) - 1,
        efix=efix,
        emode=EnergyMode(g("emode")),
        en=sc.array(dims=["energy_transfer"], values=en, unit="meV"),
        psi=sc.scalar(g("psi"), unit=angle_unit),
        u=sc.vector(_get_struct_field(struct, "u").data),
        v=sc.vector(_get_struct_field(struct, "v").data),
        omega=sc.scalar(g("omega"), unit=angle_unit),
        dpsi=sc.scalar(g("dpsi"), unit=angle_unit),
        gl=sc.scalar(g("gl"), unit=angle_unit),
        gs=sc.scalar(g("gs"), unit=angle_unit),
    )


def _parse_unique_references_container_1_0(struct: ir.Struct) -> list[Any]:
    objects = _get_struct_field(struct, "unique_objects").data
    if len(objects) != 1:
        raise AbortParse(
            "Expected exactly one object in a 'unique_references_container', "
            f"got {len(objects)}"
        )
    return _parse_unique_objects_container_1_0(objects[0])


def _parse_unique_objects_container_1_0(struct: ir.Struct) -> list[Any]:
    objects = _get_struct_field(struct, "unique_objects").data
    idx = _get_struct_field(struct, "idx").data
    if not isinstance(idx, np.ndarray):  # it is list[ir.F64]
        idx = np.array([i.value for i in idx])
    parsed_objects = [_try_parse_block(obj) for obj in objects]
    # -1 to convert to 0-based index
    return [parsed_objects[int(i) - 1] for i in idx]


_BLOCK_PARSERS = {
    (SqwMainHeader.serial_name, SqwMainHeader.version): _parse_main_header_cl_2_0,
    (SqwDndMetadata.serial_name, SqwDndMetadata.version): _parse_dnd_metadata_1_0,
    (SqwPixelMetadata.serial_name, SqwPixelMetadata.version): _parse_pix_metadata_1_0,
    (
        SqwIXNullInstrument.serial_name,
        SqwIXNullInstrument.version,
    ): _parse_ix_null_instrument_1_0,
    (SqwIXExperiment.serial_name, SqwIXExperiment.version): _parse_ix_experiment_3_0,
    (SqwIXSample.serial_name, SqwIXSample.version): _parse_ix_sample_0_0,
    (SqwIXSource.serial_name, SqwIXSource.version): _parse_ix_source_2_0,
    ("unique_references_container", 1.0): _parse_unique_references_container_1_0,
    ("unique_objects_container", 1.0): _parse_unique_objects_container_1_0,
}


def _read_pix_block(sqw_io: LowLevelSqw) -> npt.NDArray[np.float32]:
    n_rows = sqw_io.read_u32()
    n_pixels = sqw_io.read_u64()
    return sqw_io.read_array((n_rows, n_pixels), np.dtype("float32"))


def _read_dnd_block(
    sqw_io: LowLevelSqw,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.uint64]]:
    # like metadata.axes.n_bins_all_dims?
    n_dims = sqw_io.read_u32()  # u32 not u8 as normal
    shape = tuple(sqw_io.read_u32() for _ in range(n_dims))
    values = sqw_io.read_array(shape, np.dtype("float64"))
    errors = sqw_io.read_array(shape, np.dtype("float64"))
    counts = sqw_io.read_array(shape, np.dtype("uint64"))
    return values, errors, counts
