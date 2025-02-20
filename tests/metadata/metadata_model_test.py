# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc
import scippnexus as snx
from dateutil.parser import parse as parse_datetime

import scippneutron as scn
import scippneutron.data
from scippneutron import metadata


def test_measurement_from_nexus_entry() -> None:
    with snx.File(scn.data.get_path('PG3_4844_event.nxs')) as f:
        experiment = metadata.Measurement.from_nexus_entry(f['entry'])
    assert experiment.title == 'diamond cw0.533 4.22e12 60Hz [10x30]'
    assert experiment.run_number == '4844'
    assert experiment.experiment_id == 'IPTS-2767'
    assert experiment.start_time == parse_datetime('2011-08-12T11:50:17-04:00')
    assert experiment.end_time == parse_datetime('2011-08-12T13:22:05-04:00')
    assert experiment.experiment_doi is None


def test_measurement_from_individual_variables() -> None:
    experiment = metadata.Measurement(
        title=sc.scalar('The title'),
        run_number='12b',
        experiment_id=sc.scalar('EXP-1', unit=''),
    )
    assert experiment.title == 'The title'
    assert experiment.run_number == '12b'
    assert experiment.experiment_id == 'EXP-1'
    assert experiment.experiment_doi is None


def test_beamline_from_nexus_entry() -> None:
    with snx.File(scn.data.get_path('PG3_4844_event.nxs')) as f:
        beamline = metadata.Beamline.from_nexus_entry(f['entry'])
    assert beamline.name == 'POWGEN'
    assert beamline.facility is None
    assert beamline.site is None
    assert beamline.revision is None


def test_software_from_from_package_metadata_first_party() -> None:
    software = metadata.Software.from_package_metadata('scippneutron')
    expected = metadata.Software(
        name='scippneutron',
        version=scn.__version__,
        url='https://github.com/scipp/scippneutron',
        doi=None,  # Cannot be deduced
    )
    assert software == expected


def test_software_from_from_package_metadata_third_party() -> None:
    software = metadata.Software.from_package_metadata('scipp')
    assert software.name == 'scipp'
    assert software.version == sc.__version__
    assert software.url in (
        'https://github.com/scipp/scipp',  # properly deduced
        None,  # fallback for our conda packages
    )
    assert software.doi is None


def test_software_from_from_package_metadata_fails_when_package_not_installed() -> None:
    with pytest.raises(ModuleNotFoundError):
        metadata.Software.from_package_metadata('not-a-package')
