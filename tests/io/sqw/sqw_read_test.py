# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from datetime import datetime
from io import BytesIO
from pathlib import Path

import pytest
import scipp as sc
import scipp.testing

from scippneutron.io.sqw import Byteorder, EnergyMode, Sqw, SqwFileHeader, SqwFileType


@pytest.fixture(params=["file", "buffer"])
def intact_v4_sqw(request: pytest.FixtureRequest) -> Path | BytesIO:
    from scippneutron.data import get_path

    path = Path(get_path("horace_sqw_4d.sqw"))
    if request.param == "buffer":
        return BytesIO(path.read_bytes())
    return path


def test_detects_byteorder_little_endian() -> None:
    buf = BytesIO(
        b"\x06\x00\x00\x00"
        b"horace"
        b"\x00\x00\x00\x00\x00\x00\x10\x40"
        b"\x01\x00\x00\x00"
        b"\x04\x00\x00\x00"
    )
    with Sqw.open(buf) as sqw:
        assert sqw.byteorder == Byteorder.little


def test_detects_byteorder_big_endian() -> None:
    buf = BytesIO(
        b"\x00\x00\x00\x06"
        b"horace"
        b"\x40\x10\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x01"
        b"\x00\x00\x00\x04"
    )
    with Sqw.open(buf) as sqw:
        assert sqw.byteorder == Byteorder.big


def test_open_file_header_little_endian() -> None:
    buf = BytesIO(
        b"\x06\x00\x00\x00"
        b"horace"
        b"\x00\x00\x00\x00\x00\x00\x10\x40"
        b"\x01\x00\x00\x00"
        b"\x04\x00\x00\x00"
    )
    expected = SqwFileHeader(
        prog_name="horace",
        prog_version=4.0,
        sqw_type=SqwFileType.SQW,
        n_dims=4,
    )
    with Sqw.open(buf) as sqw:
        assert sqw.file_header == expected


def test_open_file_header_big_endian() -> None:
    buf = BytesIO(
        b"\x00\x00\x00\x06"
        b"horace"
        b"\x40\x10\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x01"
        b"\x00\x00\x00\x04"
    )
    expected = SqwFileHeader(
        prog_name="horace",
        prog_version=4.0,
        sqw_type=SqwFileType.SQW,
        n_dims=4,
    )
    with Sqw.open(buf) as sqw:
        assert sqw.file_header == expected


def test_open_flags_wrong_prog_name() -> None:
    buf = BytesIO(
        b"\x07\x00\x00\x00"
        b"sqomega"
        b"\x00\x00\x00\x00\x00\x00\x10\x40"
        b"\x01\x00\x00\x00"
        b"\x04\x00\x00\x00"
    )
    expected = SqwFileHeader(
        prog_name="sqomega",
        prog_version=4.0,
        sqw_type=SqwFileType.SQW,
        n_dims=4,
    )
    with pytest.warns(UserWarning, match="SQW program not supported"):
        with Sqw.open(buf) as sqw:
            assert sqw.file_header == expected


def test_open_flags_wrong_prog_version() -> None:
    buf = BytesIO(
        b"\x06\x00\x00\x00"
        b"horace"
        b"\x00\x00\x00\x00\x00\x00\x20\x40"
        b"\x01\x00\x00\x00"
        b"\x04\x00\x00\x00"
    )
    expected = SqwFileHeader(
        prog_name="horace",
        prog_version=8.0,
        sqw_type=SqwFileType.SQW,
        n_dims=4,
    )
    with pytest.warns(UserWarning, match="SQW program not supported"):
        with Sqw.open(buf) as sqw:
            assert sqw.file_header == expected


def test_read_data_block_raises_when_given_tuple_and_str(
    intact_v4_sqw: Path | BytesIO,
) -> None:
    with Sqw.open(intact_v4_sqw) as sqw:
        with pytest.raises(TypeError):
            sqw.read_data_block(("", "main_header"), "extra")


def test_read_data_block_raises_when_given_only_one_str(
    intact_v4_sqw: Path | BytesIO,
) -> None:
    with Sqw.open(intact_v4_sqw) as sqw:
        with pytest.raises(TypeError):
            sqw.read_data_block("main_header")


def test_read_main_header(intact_v4_sqw: Path | BytesIO) -> None:
    with Sqw.open(intact_v4_sqw) as sqw:
        main_header = sqw.read_data_block(("", "main_header"))
    assert main_header.version == 2.0
    assert main_header.title == ""
    assert type(main_header.nfiles) is int  # because it is encoded as f64 in file
    assert main_header.nfiles == 23
    # TODO can we encode a timezone? How does horace react?
    assert main_header.creation_date == datetime(2024, 3, 21, 21, 16, 56)  # noqa: DTZ001


def test_read_expdata(intact_v4_sqw: Path | BytesIO) -> None:
    with Sqw.open(intact_v4_sqw) as sqw:
        main_header = sqw.read_data_block("", "main_header")
        expdata = sqw.read_data_block("experiment_info", "expdata")
    assert len(expdata) == main_header.nfiles
    assert expdata[0].emode == EnergyMode.direct
    # Numbers copied from HORACE:
    sc.testing.assert_identical(
        expdata[0].psi, sc.scalar(-0.03490658476948738, unit='rad')
    )
    sc.testing.assert_identical(
        expdata[0].u,
        sc.vector([0.9957987070083618, 0.002324522938579321, -0.004358167294412851]),
    )


def test_read_sample(intact_v4_sqw: Path | BytesIO) -> None:
    with Sqw.open(intact_v4_sqw) as sqw:
        main_header = sqw.read_data_block("", "main_header")
        samples = sqw.read_data_block("experiment_info", "samples")
    assert len(samples) == main_header.nfiles
    assert all(sample == samples[0] for sample in samples[1:])
    sc.testing.assert_identical(
        samples[0].lattice_spacing,
        sc.vector([2.8579773902893066] * 3, unit="1/angstrom"),
    )
    sc.testing.assert_identical(
        samples[0].lattice_angle, sc.vector([90, 90, 90], unit="deg")
    )
