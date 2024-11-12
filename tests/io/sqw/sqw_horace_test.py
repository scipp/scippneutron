# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipp as sc
from dateutil.parser import parse as parse_datetime

from scippneutron.io.sqw import (
    EnergyMode,
    Sqw,
    SqwDndMetadata,
    SqwIXExperiment,
    SqwIXNullInstrument,
    SqwIXSample,
    SqwIXSource,
    SqwLineAxes,
    SqwLineProj,
)

pace_neutrons = pytest.importorskip("pace_neutrons")


@pytest.fixture(scope="module")
def matlab() -> Any:
    try:
        return pace_neutrons.Matlab()
    except RuntimeError as err:
        if err.args[0].startswith("No supported MATLAB"):
            pytest.skip("Not MATLAB runtime found")
        else:
            raise


@pytest.fixture
def dnd_metadata() -> SqwDndMetadata:
    return SqwDndMetadata(
        axes=SqwLineAxes(
            title="My Axes",
            label=["u1", "u2", "u3", "u4"],
            img_scales=[
                sc.scalar(1.0, unit="1/angstrom"),
                sc.scalar(1.0, unit="1/angstrom"),
                sc.scalar(1.0, unit="1/angstrom"),
                sc.scalar(1.0, unit="meV"),
            ],
            img_range=[
                sc.array(dims=["range"], values=[0.0, 1.0], unit="1/angstrom"),
                sc.array(dims=["range"], values=[0.0, 1.0], unit="1/angstrom"),
                sc.array(dims=["range"], values=[0.0, 1.0], unit="1/angstrom"),
                sc.array(dims=["range"], values=[0.0, 1.0], unit="meV"),
            ],
            # must be > 1
            n_bins_all_dims=sc.array(dims=["axis"], values=[2, 2, 2, 2], unit=None),
            single_bin_defines_iax=sc.array(dims=["axis"], values=[True] * 4),
            dax=sc.arange("axis", 4, unit=None),
            offset=[
                sc.scalar(0.0, unit="1/angstrom"),
                sc.scalar(0.0, unit="1/angstrom"),
                sc.scalar(0.0, unit="1/angstrom"),
                sc.scalar(0.0, unit="meV"),
            ],
            changes_aspect_ratio=True,
            filename="dnd_axes",
            filepath="/dnd",
        ),
        proj=SqwLineProj(
            title="My Projection",
            lattice_spacing=sc.vector([2.86, 2.86, 2.86], unit="1/angstrom"),
            lattice_angle=sc.vector([90.0, 90.0, 90.0], unit="deg"),
            offset=[
                sc.scalar(0.0, unit="1/angstrom"),
                sc.scalar(0.0, unit="1/angstrom"),
                sc.scalar(0.0, unit="1/angstrom"),
                sc.scalar(0.0, unit="meV"),
            ],
            label=["u1", "u2", "u3", "u4"],
            u=sc.vector([1.0, 0.0, 0.0], unit="1/angstrom"),
            v=sc.vector([0.0, 1.0, 0.0], unit="1/angstrom"),
            w=None,
            non_orthogonal=False,
            type="aaa",
        ),
    )


@pytest.fixture
def null_instrument() -> SqwIXNullInstrument:
    return SqwIXNullInstrument(
        name="Custom Instrument",
        source=SqwIXSource(
            name="My Source",
            target_name="The target",
            frequency=sc.scalar(13.4, unit="MHz"),
        ),
    )


@pytest.fixture
def sample() -> SqwIXSample:
    return SqwIXSample(
        name="Vibranium",
        lattice_spacing=sc.vector([2.86, 2.86, 2.86], unit="1/angstrom"),
        lattice_angle=sc.vector([90.0, 90.0, 90.0], unit="deg"),
    )


@pytest.fixture
def experiment_template() -> SqwIXExperiment:
    return SqwIXExperiment(
        run_id=0,
        efix=sc.scalar(1.2, unit="meV"),
        emode=EnergyMode.direct,
        en=sc.array(dims=["energy_transfer"], values=[-0.1, 0.3, 0.5], unit="meV"),
        psi=sc.scalar(0.4, unit="rad"),
        u=sc.vector([1.0, 0.0, 0.0], unit="1/angstrom"),
        v=sc.vector([0.0, 1.0, 0.0], unit="1/angstrom"),
        omega=sc.scalar(-0.01, unit="rad"),
        dpsi=sc.scalar(0.0, unit="rad"),
        gl=sc.scalar(1.2, unit="rad"),
        gs=sc.scalar(0.6, unit="rad"),
        filename="experiment1.nxspe",
        filepath="/data",
    )


def test_horace_roundtrip_main_header(
    matlab: Any,
    dnd_metadata: SqwDndMetadata,
    null_instrument: SqwIXNullInstrument,
    sample: SqwIXSample,
    experiment_template: SqwIXExperiment,
    tmp_path: Path,
) -> None:
    path = tmp_path / "roundtrip_main_header.sqw"
    with (
        Sqw.build(path, title="Minimal test file")
        .add_default_instrument(null_instrument)
        .add_default_sample(sample)
        .add_empty_dnd_data(dnd_metadata)
        .register_pixel_data(n_pixels=1, n_dims=4, experiments=[experiment_template])
        .create()
    ) as sqw:
        sqw.write_pixel_data(np.zeros((1, 9)), run=0)

    loaded_file = matlab.read_horace(os.fspath(path))
    main_header = loaded_file.main_header
    assert main_header.filename == path.name
    assert main_header.title == "Minimal test file"
    assert main_header.nfiles == 1
    assert (
        parse_datetime(main_header.creation_date).astimezone(tz=timezone.utc)
        - datetime.now(tz=timezone.utc)
    ) < timedelta(seconds=5)


def test_horace_roundtrip_null_instruments(
    matlab: Any,
    dnd_metadata: SqwDndMetadata,
    null_instrument: SqwIXNullInstrument,
    sample: SqwIXSample,
    experiment_template: SqwIXExperiment,
    tmp_path: Path,
) -> None:
    path = tmp_path / "roundtrip_null_instrument.sqw"
    with (
        Sqw.build(path)
        .add_default_instrument(null_instrument)
        .add_default_sample(sample)
        .add_empty_dnd_data(dnd_metadata)
        .register_pixel_data(n_pixels=1, n_dims=4, experiments=[experiment_template])
        .create()
    ) as sqw:
        sqw.write_pixel_data(np.zeros((1, 9)), run=0)

    loaded_file = matlab.read_horace(os.fspath(path))
    loaded_instruments = loaded_file.experiment_info.instruments.unique_objects
    assert np.array(loaded_instruments.n_objects).squeeze() == 1
    loaded = loaded_instruments[0]
    assert loaded.name == null_instrument.name
    assert loaded.source.name == null_instrument.source.name
    assert loaded.source.target_name == null_instrument.source.target_name
    assert loaded.source.frequency == null_instrument.source.frequency.value


def test_horace_roundtrip_sample(
    matlab: Any,
    dnd_metadata: SqwDndMetadata,
    null_instrument: SqwIXNullInstrument,
    sample: SqwIXSample,
    experiment_template: SqwIXExperiment,
    tmp_path: Path,
) -> None:
    path = tmp_path / "roundtrip_sample.sqw"
    with (
        Sqw.build(path)
        .add_default_instrument(null_instrument)
        .add_default_sample(sample)
        .add_empty_dnd_data(dnd_metadata)
        .register_pixel_data(n_pixels=1, n_dims=4, experiments=[experiment_template])
        .create()
    ) as sqw:
        sqw.write_pixel_data(np.zeros((1, 9)), run=0)

    loaded_file = matlab.read_horace(os.fspath(path))
    loaded_samples = loaded_file.experiment_info.samples.unique_objects
    assert loaded_samples.n_objects.squeeze() == 1
    loaded = loaded_samples[0]
    assert loaded.name == sample.name
    np.testing.assert_equal(
        loaded.alatt.squeeze(), sample.lattice_spacing.to(unit="1/angstrom").values
    )
    np.testing.assert_equal(
        loaded.angdeg.squeeze(), sample.lattice_angle.to(unit="deg").values
    )


def test_horace_roundtrip_experiment(
    matlab: Any,
    dnd_metadata: SqwDndMetadata,
    null_instrument: SqwIXNullInstrument,
    sample: SqwIXSample,
    experiment_template: SqwIXExperiment,
    tmp_path: Path,
) -> None:
    path = tmp_path / "roundtrip_experiment.sqw"
    with (
        Sqw.build(path)
        .add_default_instrument(null_instrument)
        .add_default_sample(sample)
        .add_empty_dnd_data(dnd_metadata)
        .register_pixel_data(n_pixels=1, n_dims=4, experiments=[experiment_template])
        .create()
    ) as sqw:
        sqw.write_pixel_data(np.zeros((1, 9)), run=0)

    loaded_file = matlab.read_horace(os.fspath(path))
    loaded_experiments = loaded_file.experiment_info.expdata
    assert matlab.numel(loaded_experiments).squeeze() == 1
    loaded = loaded_experiments[0]
    expected = experiment_template

    assert loaded.run_id.squeeze() == expected.run_id + 1
    np.testing.assert_equal(loaded.efix, expected.efix.to(unit="meV").value)
    assert loaded.emode == expected.emode.value
    np.testing.assert_equal(loaded.en.squeeze(), expected.en.to(unit="meV").values)
    np.testing.assert_equal(loaded.psi.squeeze(), expected.psi.to(unit="rad").value)
    np.testing.assert_equal(loaded.u.squeeze(), expected.u.to(unit="1/angstrom").values)
    np.testing.assert_equal(loaded.v.squeeze(), expected.v.to(unit="1/angstrom").values)
    np.testing.assert_equal(loaded.omega.squeeze(), expected.omega.to(unit="rad").value)
    np.testing.assert_equal(loaded.dpsi.squeeze(), expected.dpsi.to(unit="rad").value)
    np.testing.assert_equal(loaded.gl.squeeze(), expected.gl.to(unit="rad").value)
    np.testing.assert_equal(loaded.gs.squeeze(), expected.gs.to(unit="rad").value)
    assert loaded.filename == expected.filename
    assert loaded.filepath == expected.filepath


def test_horace_roundtrip_pixels_single_experiment(
    matlab: Any,
    dnd_metadata: SqwDndMetadata,
    null_instrument: SqwIXNullInstrument,
    sample: SqwIXSample,
    experiment_template: SqwIXExperiment,
    tmp_path: Path,
) -> None:
    path = tmp_path / "roundtrip_pixels_single_experiment.sqw"
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

    with (
        Sqw.build(path, title="Pixel test file")
        .add_default_instrument(null_instrument)
        .add_default_sample(sample)
        .add_empty_dnd_data(dnd_metadata)
        .register_pixel_data(
            n_pixels=n_pixels, n_dims=4, experiments=[experiment_template]
        )
        .create()
    ) as sqw:
        sqw.write_pixel_data(
            np.c_[u1, u2, u3, u4, irun, idet, ien, values, variances], run=0
        )

    loaded = matlab.read_horace(os.fspath(path))
    np.testing.assert_equal(loaded.pix.u1.squeeze(), u1)
    np.testing.assert_equal(loaded.pix.u2.squeeze(), u2)
    np.testing.assert_equal(loaded.pix.u3.squeeze(), u3)
    np.testing.assert_equal(loaded.pix.dE.squeeze(), u4)
    np.testing.assert_equal(loaded.pix.run_idx.squeeze(), irun)
    np.testing.assert_equal(loaded.pix.detector_idx.squeeze(), idet)
    np.testing.assert_equal(loaded.pix.energy_idx.squeeze(), ien)
    np.testing.assert_equal(loaded.pix.signal.squeeze(), values)
    np.testing.assert_equal(loaded.pix.variance.squeeze(), variances)


def test_horace_roundtrip_pixels_two_experiments(
    matlab: Any,
    dnd_metadata: SqwDndMetadata,
    null_instrument: SqwIXNullInstrument,
    sample: SqwIXSample,
    experiment_template: SqwIXExperiment,
    tmp_path: Path,
) -> None:
    path = tmp_path / "roundtrip_pixels_two_experiments.sqw"
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

    experiments = [
        replace(experiment_template, run_id=0, filename="experiment_1.nxspe"),
        replace(experiment_template, run_id=1, filename="experiment_2.nxspe"),
    ]

    with (
        Sqw.build(path, title="Pixel test file")
        .add_default_instrument(null_instrument)
        .add_default_sample(sample)
        .add_empty_dnd_data(dnd_metadata)
        .register_pixel_data(
            n_pixels=n_pixels * len(experiments), n_dims=4, experiments=experiments
        )
        .create()
    ) as sqw:
        sqw.write_pixel_data(
            np.c_[u1, u2, u3, u4, irun, idet, ien, values, variances], run=0
        )
        sqw.write_pixel_data(
            np.c_[u1, u2, u3, u4, irun, idet, ien, values, variances] + 1000, run=1
        )

    loaded = matlab.read_horace(os.fspath(path))
    np.testing.assert_equal(loaded.pix.u1.squeeze(), np.r_[u1, u1 + 1000])
    np.testing.assert_equal(loaded.pix.u2.squeeze(), np.r_[u2, u2 + 1000])
    np.testing.assert_equal(loaded.pix.u3.squeeze(), np.r_[u3, u3 + 1000])
    np.testing.assert_equal(loaded.pix.dE.squeeze(), np.r_[u4, u4 + 1000])
    np.testing.assert_equal(loaded.pix.run_idx.squeeze(), np.r_[irun, irun + 1000])
    np.testing.assert_equal(loaded.pix.detector_idx.squeeze(), np.r_[idet, idet + 1000])
    np.testing.assert_equal(loaded.pix.energy_idx.squeeze(), np.r_[ien, ien + 1000])
    np.testing.assert_equal(loaded.pix.signal.squeeze(), np.r_[values, values + 1000])
    np.testing.assert_equal(
        loaded.pix.variance.squeeze(), np.r_[variances, variances + 1000]
    )


# TODO indirect
# TODO ndims
