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
    except RuntimeError as e:
        if "No supported MATLAB" in e.args[0]:
            pytest.skip("MATLAB is unavailable")
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
            lattice_spacing=sc.vector([2.86, 2.86, 2.86], unit="angstrom"),
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
        lattice_spacing=sc.vector([2.86, 2.86, 2.86], unit="angstrom"),
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


@pytest.fixture
def pixel_data() -> sc.DataArray:
    rng = np.random.default_rng(9293)
    n_pixels = 76
    # Chosen numbers can all be represented in float32 to allow exact comparisons.
    return sc.DataArray(
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


def test_horace_roundtrip_main_header(
    matlab: Any,
    dnd_metadata: SqwDndMetadata,
    null_instrument: SqwIXNullInstrument,
    sample: SqwIXSample,
    experiment_template: SqwIXExperiment,
    pixel_data: sc.DataArray,
    tmp_path: Path,
) -> None:
    path = tmp_path / "roundtrip_main_header.sqw"
    (
        Sqw.build(path, title="Minimal test file")
        .add_default_instrument(null_instrument)
        .add_default_sample(sample)
        .add_empty_dnd_data(dnd_metadata)
        .add_pixel_data(pixel_data, experiments=[experiment_template])
        .create()
    )

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
    pixel_data: sc.DataArray,
    tmp_path: Path,
) -> None:
    path = tmp_path / "roundtrip_null_instrument.sqw"
    (
        Sqw.build(path)
        .add_default_instrument(null_instrument)
        .add_default_sample(sample)
        .add_empty_dnd_data(dnd_metadata)
        .add_pixel_data(pixel_data, experiments=[experiment_template])
        .create()
    )

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
    pixel_data: sc.DataArray,
    tmp_path: Path,
) -> None:
    path = tmp_path / "roundtrip_sample.sqw"
    (
        Sqw.build(path)
        .add_default_instrument(null_instrument)
        .add_default_sample(sample)
        .add_empty_dnd_data(dnd_metadata)
        .add_pixel_data(pixel_data, experiments=[experiment_template])
        .create()
    )

    loaded_file = matlab.read_horace(os.fspath(path))
    loaded_samples = loaded_file.experiment_info.samples.unique_objects
    assert loaded_samples.n_objects.squeeze() == 1
    loaded = loaded_samples[0]
    assert loaded.name == sample.name
    np.testing.assert_equal(
        loaded.alatt.squeeze(), sample.lattice_spacing.to(unit="angstrom").values
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
    pixel_data: sc.DataArray,
    tmp_path: Path,
) -> None:
    path = tmp_path / "roundtrip_experiment.sqw"
    (
        Sqw.build(path)
        .add_default_instrument(null_instrument)
        .add_default_sample(sample)
        .add_empty_dnd_data(dnd_metadata)
        .add_pixel_data(pixel_data, experiments=[experiment_template])
        .create()
    )

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


def test_horace_roundtrip_experiment_indirect(
    matlab: Any,
    dnd_metadata: SqwDndMetadata,
    null_instrument: SqwIXNullInstrument,
    sample: SqwIXSample,
    pixel_data: sc.DataArray,
    tmp_path: Path,
) -> None:
    experiment_template = SqwIXExperiment(
        run_id=0,
        efix=sc.array(dims=['detector'], values=[0.5, 0.6, 0.8, 0.9, 1.1], unit="meV"),
        emode=EnergyMode.direct,
        en=sc.array(
            dims=["detector", "energy_transfer"],
            values=[[-0.1, 0.3], [-0.2, 0.2], [0.0, 0.6], [0.1, 0.7], [0.3, 0.7]],
            unit="meV",
        ),
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

    path = tmp_path / "roundtrip_experiment_indirect.sqw"
    (
        Sqw.build(path)
        .add_default_instrument(null_instrument)
        .add_default_sample(sample)
        .add_empty_dnd_data(dnd_metadata)
        .add_pixel_data(pixel_data, experiments=[experiment_template])
        .create()
    )

    loaded_file = matlab.read_horace(os.fspath(path))
    loaded_experiments = loaded_file.experiment_info.expdata
    assert matlab.numel(loaded_experiments).squeeze() == 1
    loaded = loaded_experiments[0]
    expected = experiment_template

    assert loaded.run_id.squeeze() == expected.run_id + 1
    np.testing.assert_equal(loaded.efix.squeeze(), expected.efix.to(unit="meV").values)
    assert loaded.emode == expected.emode.value
    np.testing.assert_equal(
        loaded.en.reshape(expected.en.shape), expected.en.to(unit="meV").values
    )
    np.testing.assert_equal(loaded.psi.squeeze(), expected.psi.to(unit="rad").value)
    np.testing.assert_equal(loaded.u.squeeze(), expected.u.to(unit="1/angstrom").values)
    np.testing.assert_equal(loaded.v.squeeze(), expected.v.to(unit="1/angstrom").values)
    np.testing.assert_equal(loaded.omega.squeeze(), expected.omega.to(unit="rad").value)
    np.testing.assert_equal(loaded.dpsi.squeeze(), expected.dpsi.to(unit="rad").value)
    np.testing.assert_equal(loaded.gl.squeeze(), expected.gl.to(unit="rad").value)
    np.testing.assert_equal(loaded.gs.squeeze(), expected.gs.to(unit="rad").value)
    assert loaded.filename == expected.filename
    assert loaded.filepath == expected.filepath


def test_horace_roundtrip_pixels(
    matlab: Any,
    dnd_metadata: SqwDndMetadata,
    null_instrument: SqwIXNullInstrument,
    sample: SqwIXSample,
    experiment_template: SqwIXExperiment,
    pixel_data: sc.DataArray,
    tmp_path: Path,
) -> None:
    path = tmp_path / "roundtrip_pixels.sqw"

    experiments = [
        replace(experiment_template, run_id=0, filename="experiment_1.nxspe"),
        replace(experiment_template, run_id=1, filename="experiment_2.nxspe"),
    ]
    rng = np.random.default_rng(526)
    pixel_data.coords['irun'] = sc.array(
        dims=['obs'], values=rng.integers(0, 2, len(pixel_data)), unit=None
    )

    (
        Sqw.build(path, title="Pixel test file")
        .add_default_instrument(null_instrument)
        .add_default_sample(sample)
        .add_empty_dnd_data(dnd_metadata)
        .add_pixel_data(pixel_data, experiments=experiments)
        .create()
    )

    loaded = matlab.read_horace(os.fspath(path))
    np.testing.assert_equal(loaded.pix.u1.squeeze(), pixel_data.coords['u1'].values)
    np.testing.assert_equal(loaded.pix.u2.squeeze(), pixel_data.coords['u2'].values)
    np.testing.assert_equal(loaded.pix.u3.squeeze(), pixel_data.coords['u3'].values)
    np.testing.assert_equal(loaded.pix.dE.squeeze(), pixel_data.coords['u4'].values)
    np.testing.assert_equal(
        loaded.pix.run_idx.squeeze(), pixel_data.coords['irun'].values
    )
    np.testing.assert_equal(
        loaded.pix.detector_idx.squeeze(), pixel_data.coords['idet'].values
    )
    np.testing.assert_equal(
        loaded.pix.energy_idx.squeeze(), pixel_data.coords['ien'].values
    )
    np.testing.assert_equal(loaded.pix.signal.squeeze(), pixel_data.values)
    np.testing.assert_equal(loaded.pix.variance.squeeze(), pixel_data.variances)
