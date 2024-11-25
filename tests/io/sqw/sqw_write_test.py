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
    builder.create()
    buffer.rewind()

    with Sqw.open(buffer.get()) as sqw:
        assert sqw.byteorder.value == sys.byteorder


def test_create_sets_byteorder_little(buffer: _BytesBuffer | _PathBuffer) -> None:
    builder = Sqw.build(buffer.get(), byteorder="little")
    builder.create()
    buffer.rewind()

    with Sqw.open(buffer.get()) as sqw:
        assert sqw.byteorder == Byteorder.little


def test_create_sets_byteorder_big(buffer: _BytesBuffer | _PathBuffer) -> None:
    builder = Sqw.build(buffer.get(), byteorder="big")
    builder.create()
    buffer.rewind()

    with Sqw.open(buffer.get()) as sqw:
        assert sqw.byteorder == Byteorder.big


def test_create_writes_file_header_little_endian(
    buffer: _BytesBuffer | _PathBuffer,
) -> None:
    builder = Sqw.build(buffer.get(), byteorder="little")
    builder.create()

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
    builder.create()

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
    builder.create()
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

    n_pixels = 7
    pixels = sc.DataArray(
        sc.zeros(sizes={'obs': n_pixels}, with_variances=True, unit='count'),
        coords={
            'u1': sc.zeros(sizes={'obs': n_pixels}, unit='1/Å'),
            'u2': sc.zeros(sizes={'obs': n_pixels}, unit='1/Å'),
            'u3': sc.zeros(sizes={'obs': n_pixels}, unit='1/Å'),
            'u4': sc.zeros(sizes={'obs': n_pixels}, unit='meV'),
            'idet': sc.zeros(sizes={'obs': n_pixels}, dtype=int, unit=None),
            'irun': sc.zeros(sizes={'obs': n_pixels}, dtype=int, unit=None),
            'ien': sc.zeros(sizes={'obs': n_pixels}, dtype=int, unit=None),
        },
    )

    builder = Sqw.build(buffer.get(), byteorder=byteorder)
    builder = builder.add_pixel_data(pixels, experiments=experiments)
    builder.create()
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
    rng = np.random.default_rng(1732)
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

    # Chosen numbers can all be represented in float32 to allow exact comparisons.
    n_pixels = 7
    original = sc.DataArray(
        sc.array(
            dims=['obs'],
            values=rng.uniform(0, 100, n_pixels).astype('float32'),
            variances=rng.uniform(0.1, 1, n_pixels).astype('float32'),
            unit='count',
        ),
        coords={
            'idet': sc.arange('obs', 0, n_pixels, unit=None).astype(int) // sc.index(3),
            'irun': sc.arange('obs', 0, n_pixels, unit=None).astype(int) // sc.index(2),
            'ien': sc.arange('obs', 0, 2 * n_pixels, 2, unit=None).astype(int)
            // sc.index(10),
            'u1': sc.arange('obs', 0.0, n_pixels + 0.0, unit='1/Å'),
            'u2': sc.arange('obs', 1.0, n_pixels + 1.0, unit='1/Å'),
            'u3': sc.arange('obs', 2.0, n_pixels + 2.0, unit='1/Å'),
            'u4': sc.arange('obs', n_pixels, unit='meV') * 2,
        },
    )
    builder = Sqw.build(buffer.get(), byteorder=byteorder)
    builder = builder.add_pixel_data(original, experiments=experiments)
    builder.create()
    buffer.rewind()

    with Sqw.open(buffer.get()) as sqw:
        loaded = sqw.read_data_block(("pix", "data_wrap"))

    np.testing.assert_equal(loaded[:, 0], original.coords['u1'].values)
    np.testing.assert_equal(loaded[:, 1], original.coords['u2'].values)
    np.testing.assert_equal(loaded[:, 2], original.coords['u3'].values)
    np.testing.assert_equal(loaded[:, 3], original.coords['u4'].values)
    np.testing.assert_equal(loaded[:, 4], original.coords['irun'].values)
    np.testing.assert_equal(loaded[:, 5], original.coords['idet'].values)
    np.testing.assert_equal(loaded[:, 6], original.coords['ien'].values)
    np.testing.assert_equal(loaded[:, 7], original.values)
    np.testing.assert_equal(loaded[:, 8], original.variances)


@pytest.mark.parametrize("byteorder", ["native", "little", "big"])
def test_writes_pixel_data_chunked(
    byteorder: Literal["native", "little", "big"], buffer: _BytesBuffer | _PathBuffer
) -> None:
    rng = np.random.default_rng(1732)
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

    # Chosen numbers can all be represented in float32 to allow exact comparisons.
    n_pixels = 7
    original = sc.DataArray(
        sc.array(
            dims=['obs'],
            values=rng.uniform(0, 100, n_pixels).astype('float32'),
            variances=rng.uniform(0.1, 1, n_pixels).astype('float32'),
            unit='count',
        ),
        coords={
            'idet': sc.arange('obs', 0, n_pixels, unit=None).astype(int) // sc.index(3),
            'irun': sc.arange('obs', 0, n_pixels, unit=None).astype(int) // sc.index(2),
            'ien': sc.arange('obs', 0, 2 * n_pixels, 2, unit=None).astype(int)
            // sc.index(10),
            'u1': sc.arange('obs', 0.0, n_pixels + 0.0, unit='1/Å'),
            'u2': sc.arange('obs', 1.0, n_pixels + 1.0, unit='1/Å'),
            'u3': sc.arange('obs', 2.0, n_pixels + 2.0, unit='1/Å'),
            'u4': sc.arange('obs', n_pixels, unit='meV') * 2,
        },
    )
    builder = Sqw.build(buffer.get(), byteorder=byteorder)
    builder = builder.add_pixel_data(original, experiments=experiments)
    builder.create(chunk_size=n_pixels // 3)
    buffer.rewind()

    with Sqw.open(buffer.get()) as sqw:
        loaded = sqw.read_data_block(("pix", "data_wrap"))

    np.testing.assert_equal(loaded[:, 0], original.coords['u1'].values)
    np.testing.assert_equal(loaded[:, 1], original.coords['u2'].values)
    np.testing.assert_equal(loaded[:, 2], original.coords['u3'].values)
    np.testing.assert_equal(loaded[:, 3], original.coords['u4'].values)
    np.testing.assert_equal(loaded[:, 4], original.coords['irun'].values)
    np.testing.assert_equal(loaded[:, 5], original.coords['idet'].values)
    np.testing.assert_equal(loaded[:, 6], original.coords['ien'].values)
    np.testing.assert_equal(loaded[:, 7], original.values)
    np.testing.assert_equal(loaded[:, 8], original.variances)


@pytest.mark.parametrize("byteorder", ["native", "little", "big"])
def test_writes_pixel_data_convert_units(
    byteorder: Literal["native", "little", "big"], buffer: _BytesBuffer | _PathBuffer
) -> None:
    rng = np.random.default_rng(1732)
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
    original = sc.DataArray(
        sc.array(
            dims=['obs'],
            values=rng.uniform(0, 100, n_pixels),
            variances=rng.uniform(0.1, 1, n_pixels),
            unit='mega count',
        ),
        coords={
            'idet': sc.arange('obs', 0, n_pixels, unit=None).astype(int) // sc.index(3),
            'irun': sc.arange('obs', 0, n_pixels, unit=None).astype(int) // sc.index(2),
            'ien': sc.arange('obs', 0, 2 * n_pixels, 2, unit=None).astype(int)
            // sc.index(10),
            'u1': sc.arange('obs', 0.0, n_pixels + 0.0, unit='1/fm'),
            'u2': sc.arange('obs', 1.0, n_pixels + 1.0, unit='10/Å'),
            'u3': sc.arange('obs', 2.0, n_pixels + 2.0, unit='1/um'),
            'u4': sc.arange('obs', n_pixels, unit='eV') * 2,
        },
    )
    builder = Sqw.build(buffer.get(), byteorder=byteorder)
    builder = builder.add_pixel_data(original, experiments=experiments)
    builder.create()
    buffer.rewind()

    with Sqw.open(buffer.get()) as sqw:
        loaded = sqw.read_data_block(("pix", "data_wrap"))

    np.testing.assert_allclose(
        loaded[:, 0], original.coords['u1'].to(unit='1/Å').values
    )
    np.testing.assert_allclose(
        loaded[:, 1], original.coords['u2'].to(unit='1/Å').values
    )
    np.testing.assert_allclose(
        loaded[:, 2], original.coords['u3'].to(unit='1/Å').values
    )
    np.testing.assert_allclose(
        loaded[:, 3], original.coords['u4'].to(unit='meV').values
    )
    np.testing.assert_equal(loaded[:, 4], original.coords['irun'].values)
    np.testing.assert_equal(loaded[:, 5], original.coords['idet'].values)
    np.testing.assert_equal(loaded[:, 6], original.coords['ien'].values)
    np.testing.assert_allclose(loaded[:, 7], original.to(unit='count').values)
    np.testing.assert_allclose(loaded[:, 8], original.to(unit='count').variances)


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
    builder.create()
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
