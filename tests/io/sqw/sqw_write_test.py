# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import dataclasses
import sys
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import scipp as sc
import scipp.testing

from scippneutron.io.sqw import (
    Byteorder,
    EnergyMode,
    Sqw,
    SqwDndMetadata,
    SqwIXExperiment,
    SqwLineAxes,
    SqwLineProj,
)


class _PathBuffer:
    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self) -> Path:
        return self.path

    def rewind(self) -> None:
        pass

    def read(self, n: int) -> bytes:
        with self.path.open("rb") as file:
            return file.read(n)


class _BytesBuffer:
    def __init__(self) -> None:
        self.buffer = BytesIO()

    def get(self) -> BytesIO:
        return self.buffer

    def rewind(self) -> None:
        self.buffer.seek(0)

    def read(self, n: int) -> bytes:
        return self.buffer.read(n)


@pytest.fixture(params=["buffer", "file"])
def buffer(
    request: pytest.FixtureRequest, tmp_path: Path
) -> _BytesBuffer | _PathBuffer:
    match request.param:
        case "buffer":
            return _BytesBuffer()
        case "file":
            return _PathBuffer(tmp_path / "sqw_file.sqw")


def test_create_sets_byteorder_native(buffer: _BytesBuffer | _PathBuffer) -> None:
    builder = Sqw.build(buffer.get())
    with builder.create():
        pass
    buffer.rewind()

    with Sqw.open(buffer.get()) as sqw:
        assert sqw.byteorder.value == sys.byteorder


def test_create_sets_byteorder_little(buffer: _BytesBuffer | _PathBuffer) -> None:
    builder = Sqw.build(buffer.get(), byteorder="little")
    with builder.create():
        pass
    buffer.rewind()

    with Sqw.open(buffer.get()) as sqw:
        assert sqw.byteorder == Byteorder.little


def test_create_sets_byteorder_big(buffer: _BytesBuffer | _PathBuffer) -> None:
    builder = Sqw.build(buffer.get(), byteorder="big")
    with builder.create():
        pass
    buffer.rewind()

    with Sqw.open(buffer.get()) as sqw:
        assert sqw.byteorder == Byteorder.big


def test_create_writes_file_header_little_endian(
    buffer: _BytesBuffer | _PathBuffer,
) -> None:
    builder = Sqw.build(buffer.get(), byteorder="little")
    with builder.create():
        pass

    buffer.rewind()
    expected = (
        b"\x06\x00\x00\x00"
        b"horace"
        b"\x00\x00\x00\x00\x00\x00\x10\x40"
        b"\x01\x00\x00\x00"
        b"\x00\x00\x00\x00"
    )
    assert buffer.read(len(expected)) == expected


def test_create_writes_file_header_big_endian(
    buffer: _BytesBuffer | _PathBuffer,
) -> None:
    builder = Sqw.build(buffer.get(), byteorder="big")
    with builder.create():
        pass

    buffer.rewind()
    expected = (
        b"\x00\x00\x00\x06"
        b"horace"
        b"\x40\x10\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x01"
        b"\x00\x00\x00\x00"
    )
    assert buffer.read(len(expected)) == expected


@pytest.mark.parametrize("byteorder", ["native", "little", "big"])
def test_create_writes_main_header(
    byteorder: Literal["native", "little", "big"], buffer: _BytesBuffer | _PathBuffer
) -> None:
    builder = Sqw.build(buffer.get(), title="my title", byteorder=byteorder)
    with builder.create():
        pass
    buffer.rewind()

    with Sqw.open(buffer.get()) as sqw:
        main_header = sqw.read_data_block(("", "main_header"))
    filename = "in_memory" if isinstance(buffer, _BytesBuffer) else str(buffer.get())
    assert main_header.full_filename == filename
    assert main_header.title == "my title"
    assert main_header.nfiles == 0
    assert (main_header.creation_date - datetime.now(tz=timezone.utc)) < timedelta(
        seconds=1
    )


@pytest.mark.parametrize("byteorder", ["native", "little", "big"])
def test_register_pixel_data_writes_pix_metadata(
    byteorder: Literal["native", "little", "big"], buffer: _BytesBuffer | _PathBuffer
) -> None:
    builder = Sqw.build(buffer.get(), byteorder=byteorder)
    builder = builder.register_pixel_data(n_pixels=13, n_dims=3, experiments=[])
    with builder.create():
        pass
    buffer.rewind()

    with Sqw.open(buffer.get()) as sqw:
        pix_metadata = sqw.read_data_block(("pix", "metadata"))
    filename = "in_memory" if isinstance(buffer, _BytesBuffer) else str(buffer.get())
    assert pix_metadata.full_filename == filename
    assert pix_metadata.npix == 13
    assert pix_metadata.data_range.shape == (9, 2)


@pytest.mark.parametrize("byteorder", ["native", "little", "big"])
def test_writes_expdata(
    byteorder: Literal["native", "little", "big"], buffer: _BytesBuffer | _PathBuffer
) -> None:
    experiments = [
        SqwIXExperiment(
            run_id=0,
            efix=sc.scalar(1.2, unit="meV"),
            emode=EnergyMode.direct,
            en=sc.array(dims=["energy_transfer"], values=[3.0], unit="ueV"),
            psi=sc.scalar(1.2, unit="rad"),
            u=sc.vector([0.0, 1.0, 0.0]),
            v=sc.vector([1.0, 1.0, 0.0]),
            omega=sc.scalar(1.4, unit="rad"),
            dpsi=sc.scalar(46, unit="deg"),
            gl=sc.scalar(3, unit="rad"),
            gs=sc.scalar(-0.5, unit="rad"),
            filename="run1.nxspe",
            filepath="/data",
        ),
        SqwIXExperiment(
            run_id=2,
            efix=sc.scalar(0.16, unit="eV"),
            emode=EnergyMode.direct,
            en=sc.array(dims=["energy_transfer"], values=[2.0, 4.5], unit="meV"),
            psi=sc.scalar(-10.0, unit="deg"),
            u=sc.vector([1.0, 0.0, 0.0]),
            v=sc.vector([0.0, 1.0, 0.0]),
            omega=sc.scalar(-91, unit="deg"),
            dpsi=sc.scalar(-0.5, unit="rad"),
            gl=sc.scalar(0.0, unit="deg"),
            gs=sc.scalar(-5, unit="deg"),
            filename="run2.nxspe",
            filepath="/data",
        ),
    ]
    # The same as above but with canonical units.
    expected_experiments = [
        SqwIXExperiment(
            run_id=0,
            efix=sc.scalar(1.2, unit="meV"),
            emode=EnergyMode.direct,
            en=sc.array(dims=["energy_transfer"], values=[0.003], unit="meV"),
            psi=sc.scalar(1.2, unit="rad"),
            u=sc.vector([0.0, 1.0, 0.0]),
            v=sc.vector([1.0, 1.0, 0.0]),
            omega=sc.scalar(1.4, unit="rad"),
            dpsi=sc.scalar(46.0, unit="deg").to(unit="rad"),
            gl=sc.scalar(3.0, unit="rad"),
            gs=sc.scalar(-0.5, unit="rad"),
            filename="run1.nxspe",
            filepath="/data",
        ),
        SqwIXExperiment(
            run_id=2,
            efix=sc.scalar(160.0, unit="meV"),
            emode=EnergyMode.direct,
            en=sc.array(dims=["energy_transfer"], values=[2.0, 4.5], unit="meV"),
            psi=sc.scalar(-10.0, unit="deg").to(unit="rad"),
            u=sc.vector([1.0, 0.0, 0.0]),
            v=sc.vector([0.0, 1.0, 0.0]),
            omega=sc.scalar(-91.0, unit="deg").to(unit="rad"),
            dpsi=sc.scalar(-0.5, unit="rad"),
            gl=sc.scalar(0.0, unit="deg").to(unit="rad"),
            gs=sc.scalar(-5.0, unit="deg").to(unit="rad"),
            filename="run2.nxspe",
            filepath="/data",
        ),
    ]

    builder = Sqw.build(buffer.get(), byteorder=byteorder)
    builder = builder.register_pixel_data(
        n_pixels=13, n_dims=3, experiments=experiments
    )
    with builder.create():
        pass
    buffer.rewind()

    with Sqw.open(buffer.get()) as sqw:
        loaded_experiments = sqw.read_data_block(("experiment_info", "expdata"))

    for loaded, expected in zip(loaded_experiments, expected_experiments, strict=True):
        for field in dataclasses.fields(expected):
            sc.testing.assert_identical(
                getattr(loaded, field.name), getattr(expected, field.name)
            )


@pytest.mark.parametrize("byteorder", ["native", "little", "big"])
def test_writes_pixel_data(
    byteorder: Literal["native", "little", "big"], buffer: _BytesBuffer | _PathBuffer
) -> None:
    experiment_template = SqwIXExperiment(
        run_id=-1,
        efix=sc.scalar(1.2, unit="meV"),
        emode=EnergyMode.direct,
        en=sc.array(dims=["energy_transfer"], values=[3.0], unit="meV"),
        psi=sc.scalar(1.2, unit="rad"),
        u=sc.vector([0.0, 1.0, 0.0]),
        v=sc.vector([1.0, 1.0, 0.0]),
        omega=sc.scalar(1.4, unit="rad"),
        dpsi=sc.scalar(0.0, unit="rad"),
        gl=sc.scalar(3, unit="rad"),
        gs=sc.scalar(-0.5, unit="rad"),
        filename="",
        filepath="/data",
    )
    experiments = [
        dataclasses.replace(experiment_template, run_id=0, filename="f1"),
        dataclasses.replace(experiment_template, run_id=1, filename="f2"),
    ]

    n_pixels = 7
    # Chosen numbers can all be represented in float32 to allow exact comparisons.
    u1 = np.arange(n_pixels) + 0.0
    u2 = np.arange(n_pixels) + 1.0
    u3 = np.arange(n_pixels) + 3.0
    u4 = np.arange(n_pixels) * 2
    irun = np.full(n_pixels, 0)
    idet = (np.arange(n_pixels) / 3).astype(int)
    ien = np.full(n_pixels, 1)
    values = 20 * np.arange(n_pixels)
    variances = values / 2

    builder = Sqw.build(buffer.get(), byteorder=byteorder)
    builder = builder.register_pixel_data(
        n_pixels=n_pixels * 2, n_dims=4, experiments=experiments
    )
    with builder.create() as sqw:
        sqw.write_pixel_data(
            np.c_[u1, u2, u3, u4, irun, idet, ien, values, variances], run=0
        )
        sqw.write_pixel_data(
            np.c_[u1, u2, u3, u4, irun, idet, ien, values, variances] + 1000, run=1
        )
    buffer.rewind()

    with Sqw.open(buffer.get()) as sqw:
        loaded = sqw.read_data_block(("pix", "data_wrap"))

    np.testing.assert_equal(loaded[:, 0], np.r_[u1, u1 + 1000])
    np.testing.assert_equal(loaded[:, 1], np.r_[u2, u2 + 1000])
    np.testing.assert_equal(loaded[:, 2], np.r_[u3, u3 + 1000])
    np.testing.assert_equal(loaded[:, 3], np.r_[u4, u4 + 1000])
    np.testing.assert_equal(loaded[:, 4], np.r_[irun, irun + 1000])
    np.testing.assert_equal(loaded[:, 5], np.r_[idet, idet + 1000])
    np.testing.assert_equal(loaded[:, 6], np.r_[ien, ien + 1000])
    np.testing.assert_equal(loaded[:, 7], np.r_[values, values + 1000])
    np.testing.assert_equal(loaded[:, 8], np.r_[variances, variances + 1000])


@pytest.mark.parametrize("byteorder", ["native", "little", "big"])
def test_writes_data_metadata(
    byteorder: Literal["native", "little", "big"], buffer: _BytesBuffer | _PathBuffer
) -> None:
    metadata = SqwDndMetadata(
        axes=SqwLineAxes(
            title="test axes",
            label=["x", "y", "z", "dE"],
            img_scales=[
                1.0 / sc.Unit("angstrom"),
                2.0 / sc.Unit("2*angstrom"),
                0.5 / sc.Unit("kilo angstrom"),
                0.2 * sc.Unit("eV"),
            ],
            img_range=[
                sc.array(dims=["range"], values=[-30.0, 540.0], unit="1/(k angstrom)"),
                sc.array(dims=["range"], values=[-0.5, 6.7], unit="1/angstrom"),
                sc.array(dims=["range"], values=[-5.6, -2.4], unit="10/angstrom"),
                sc.array(dims=["range"], values=[6.0, 9.1], unit="meV"),
            ],
            n_bins_all_dims=sc.array(dims=["axis"], values=[40, 50, 40, 40], unit=None),
            single_bin_defines_iax=sc.array(
                dims=["axis"], values=[False, True, True, True]
            ),
            dax=sc.array(dims=["axis"], values=[2, 1, 0, 3], unit=None),
            offset=[
                1.0 / sc.Unit("2*angstrom"),
                50.0 / sc.Unit("milli angstrom"),
                0.0 / sc.Unit("angstrom"),
                0.0 * sc.Unit("meV"),
            ],
            changes_aspect_ratio=True,
        ),
        proj=SqwLineProj(
            lattice_spacing=sc.vector([2.1, 2.1, 2.5], unit="angstrom"),
            lattice_angle=sc.vector([np.pi / 2, np.pi / 4, np.pi / 2], unit="rad"),
            offset=[
                1.0 / sc.Unit("2*angstrom"),
                50.0 / sc.Unit("milli angstrom"),
                0.0 / sc.Unit("angstrom"),
                0.0 * sc.Unit("meV"),
            ],
            title="my projection",
            label=["x", "y", "z", "dE"],
            u=sc.vector([0.0, 1.0, 0.0], unit="1/angstrom"),
            v=sc.vector([1.0, 0.0, 0.0], unit="1/(milli angstrom)"),
            w=None,
            non_orthogonal=False,
            type="aaa",
        ),
    )
    # The same as above but with canonical units.
    expected_metadata = SqwDndMetadata(
        axes=SqwLineAxes(
            title="test axes",
            label=["x", "y", "z", "dE"],
            img_scales=[
                1.0 / sc.Unit("angstrom"),
                1.0 / sc.Unit("angstrom"),
                0.0005 / sc.Unit("angstrom"),
                200.0 * sc.Unit("meV"),
            ],
            img_range=[
                sc.array(dims=["range"], values=[-0.03, 0.540], unit="1/angstrom"),
                sc.array(dims=["range"], values=[-0.5, 6.7], unit="1/angstrom"),
                sc.array(dims=["range"], values=[-56.0, -24.0], unit="1/angstrom"),
                sc.array(dims=["range"], values=[6.0, 9.1], unit="meV"),
            ],
            n_bins_all_dims=sc.array(dims=["axis"], values=[40, 50, 40, 40], unit=None),
            single_bin_defines_iax=sc.array(
                dims=["axis"], values=[False, True, True, True]
            ),
            dax=sc.array(dims=["axis"], values=[2, 1, 0, 3], unit=None),
            offset=[
                0.5 / sc.Unit("angstrom"),
                50000.0 / sc.Unit("angstrom"),
                0.0 / sc.Unit("angstrom"),
                0.0 * sc.Unit("meV"),
            ],
            changes_aspect_ratio=True,
            filename="" if isinstance(buffer, _BytesBuffer) else buffer.get().name,
            filepath=""
            if isinstance(buffer, _BytesBuffer)
            else str(buffer.get().parent),
        ),
        proj=SqwLineProj(
            lattice_spacing=sc.vector([2.1, 2.1, 2.5], unit="angstrom"),
            lattice_angle=sc.vector([90.0, 45.0, 90.0], unit="deg"),
            offset=[
                0.5 / sc.Unit("angstrom"),
                50000.0 / sc.Unit("angstrom"),
                0.0 / sc.Unit("angstrom"),
                0.0 * sc.Unit("meV"),
            ],
            title="my projection",
            label=["x", "y", "z", "dE"],
            u=sc.vector([0.0, 1.0, 0.0], unit="1/angstrom"),
            v=sc.vector([1000.0, 0.0, 0.0], unit="1/angstrom"),
            w=None,
            non_orthogonal=False,
            type="aaa",
        ),
    )

    builder = Sqw.build(buffer.get(), byteorder=byteorder)
    builder = builder.add_empty_dnd_data(metadata)
    with builder.create():
        pass
    buffer.rewind()

    with Sqw.open(buffer.get()) as sqw:
        loaded_metadata = sqw.read_data_block(("data", "metadata"))

    loaded_axes = loaded_metadata.axes
    expected_axes = expected_metadata.axes
    for field in dataclasses.fields(loaded_axes):
        loaded = getattr(loaded_axes, field.name)
        expected = getattr(expected_axes, field.name)
        if isinstance(loaded, list):
            for a, b in zip(loaded, expected, strict=True):
                sc.testing.assert_identical(a, b)
        else:
            sc.testing.assert_identical(loaded, expected)
