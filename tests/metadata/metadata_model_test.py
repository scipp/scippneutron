# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from typing import cast

import h5py as h5
import pytest
import scipp as sc
import scippnexus as snx
from dateutil.parser import parse as parse_datetime

import scippneutron as scn
import scippneutron.data
from scippneutron import metadata


@pytest.fixture
def nxroot() -> snx.Group:
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        root = snx.Group(f, definitions=snx.base_definitions())
        root.create_class('entry', snx.NXentry)
        yield root


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


def test_person_from_nexus_group(nxroot: snx.Group) -> None:
    user_group = nxroot.create_class("user_Testo", snx.NXuser)
    user_group['name'] = "Testo Prøvegaard"
    user_group['email'] = ""  # empty to check validator bypass
    user_group['affiliation'] = 'Fakington University'
    user_group['address'] = 'Fakington University, Back Alley 3, Fakington'
    user_group['ORCID'] = 'https://orcid.org/0000-0000-0000-0001'
    user_group['role'] = 'Principal Investigator'
    # Exists in files by we cannot represent it in `Person`:
    user_group['facility_user_id'] = 'testo.faking'

    person = metadata.Person.from_nexus_user(user_group)
    assert person.name == "Testo Prøvegaard"
    assert person.email is None
    assert person.affiliation == 'Fakington University'
    assert person.orcid_id == 'https://orcid.org/0000-0000-0000-0001'
    assert person.role == 'Principal Investigator'
    assert person.address == 'Fakington University, Back Alley 3, Fakington'

    # We have no logic for deducing these from NeXus
    assert not person.corresponding
    assert person.owner


def test_person_write_to_nexus(nxroot: snx.Group) -> None:
    person = metadata.Person(
        name='Testo Prøvegaard',
        email='testo.prove@fake.uni',
        affiliation='Fakington University',
        address='Fakington University, Back Alley 3, Fakington',
        orcid_id='https://orcid.org/0000-0000-0000-0001',
        role='Principal Investigator',
    )

    nxroot['user_Testo'] = person

    assert nxroot['user_Testo'].attrs['NX_class'] == 'NXuser'
    loaded = cast(sc.DataGroup[str], nxroot['user_Testo'][()])
    assert loaded.keys() == {
        'address',
        'affiliation',
        'email',
        'name',
        'ORCID',
        'role',
    }
    assert loaded['name'] == 'Testo Prøvegaard'
    assert loaded['email'] == 'testo.prove@fake.uni'
    assert loaded['affiliation'] == 'Fakington University'
    assert loaded['address'] == 'Fakington University, Back Alley 3, Fakington'
    assert loaded['ORCID'] == 'https://orcid.org/0000-0000-0000-0001'
    assert loaded['role'] == 'Principal Investigator'


def test_person_write_to_nexus_empty_optional(nxroot: snx.Group) -> None:
    person = metadata.Person(
        name='Testo Prøvegaard',
    )

    nxroot['user_Testo'] = person

    assert nxroot['user_Testo'].attrs['NX_class'] == 'NXuser'
    loaded = cast(sc.DataGroup[str], nxroot['user_Testo'][()])
    assert loaded.keys() == {'name'}
    assert loaded['name'] == 'Testo Prøvegaard'


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


def test_software_write_to_nexus(nxroot: snx.Group) -> None:
    software = metadata.Software(
        name='scippneutron',
        version='0.1.0',
        url='https://github.com/scipp/scippneutron',
        doi='10.1007/s11224-022-02522-2',
    )

    nxroot['program'] = software

    assert nxroot['program'].attrs['NX_class'] == 'NXprogram'
    loaded = cast(sc.DataGroup[str], nxroot['program'][()])
    assert loaded.keys() == {'program'}
    assert loaded['program'] == 'scippneutron'
    prog = nxroot['program']['program']
    assert prog.attrs.keys() == {'version', 'url'}
    assert prog.attrs['version'] == '0.1.0'
    assert prog.attrs['url'] == 'https://github.com/scipp/scippneutron'


def test_source_write_to_nexus(nxroot: snx.Group) -> None:
    source = metadata.Source(
        name="Test source",
        source_type=metadata.SourceType.SpallationNeutronSource,
        probe=metadata.RadiationProbe.Neutron,
    )

    nxroot['source'] = source

    assert nxroot['source'].attrs['NX_class'] == 'NXsource'
    loaded = cast(sc.DataGroup[str], nxroot['source'][()])
    assert loaded.keys() == {'name', 'type', 'probe'}
    assert loaded['name'] == 'Test source'
    assert loaded['type'] == 'Spallation Neutron Source'
    assert loaded['probe'] == 'neutron'
