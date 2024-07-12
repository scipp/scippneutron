# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)


import scipp as sc
import scippnexus as snx
from dateutil.parser import parse as parse_datetime

import scippneutron as scn
import scippneutron.data
from scippneutron import meta


def test_experiment_from_nexus_entry() -> None:
    with snx.File(scn.data.get_path('PG3_4844_event.nxs')) as f:
        experiment = meta.Measurement.from_nexus_entry(f['entry'])
    assert experiment.title == 'diamond cw0.533 4.22e12 60Hz [10x30]'
    assert experiment.run_number == '4844'
    assert experiment.experiment_id == 'IPTS-2767'
    assert experiment.start_time == parse_datetime('2011-08-12T11:50:17-04:00')
    assert experiment.end_time == parse_datetime('2011-08-12T13:22:05-04:00')
    assert experiment.experiment_doi is None


def test_beamline_from_nexus_entry() -> None:
    with snx.File(scn.data.get_path('PG3_4844_event.nxs')) as f:
        beamline = meta.Beamline.from_nexus_entry(f['entry'])
    assert beamline.name == 'POWGEN'
    assert beamline.facility is None
    assert beamline.site is None
    assert beamline.revision is None


def test_software_from_from_package_metadata_first_party() -> None:
    software = meta.Software.from_package_metadata('scippneutron')
    expected = meta.Software(
        name='scippneutron',
        version=scn.__version__,
        url='https://github.com/scipp/scippneutron',
        doi=None,  # Cannot be deduced
    )
    assert software == expected


def test_software_from_from_package_metadata_third_party() -> None:
    software = meta.Software.from_package_metadata('scipp')
    expected = meta.Software(
        name='scipp',
        version=sc.__version__,
        url='https://github.com/scipp/scipp',
        doi=None,  # Cannot be deduced
    )
    assert software == expected
