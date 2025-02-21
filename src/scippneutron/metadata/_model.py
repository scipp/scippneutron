# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import enum
from datetime import datetime
from typing import Annotated, Any

import scipp as sc
import scippnexus as snx
from dateutil.parser import parse as parse_datetime
from pydantic import BaseModel, BeforeValidator, EmailStr

from ._orcid import ORCIDiD


def _unpack_variable(value: object) -> Any:
    """Before validator to support passing scalar scipp variables as inputs."""
    if isinstance(value, sc.Variable):
        if value.dims:
            raise ValueError("Must be a scalar")
        if value.unit and value.unit != '':
            raise ValueError("Must be dimensionless or have no unit at all")
        return value.value
    return value


class Beamline(BaseModel):
    """A beamline / instrument.

    ``name`` should be the canonical spelling of the beamline name.
    The location of the beamline is split into ``facility`` and ``site``, where
    a 'facility' is located at a 'site'. For example:

    >>> from scippneutron.metadata import Beamline
    >>> beamline = Beamline(
    ...     name='Amor',
    ...     facility='SINQ',
    ...     site='PSI',
    ... )

    If there is no separate facility and site, either omit ``site`` or use the same
    value for both ``facility`` and ``site``. For example:

    >>> beamline = Beamline(
    ...     name='ESTIA',
    ...     facility='ESS',
    ...     site='ESS',  # can be omitted
    ... )

    If the beamline has been upgraded, provide a revision to indicate
    which version of the beamline was used.
    """

    name: Annotated[str, BeforeValidator(_unpack_variable)]
    """Name of the beamline."""
    facility: Annotated[str | None, BeforeValidator(_unpack_variable)] = None
    """Facility where the beamline is located."""
    site: Annotated[str | None, BeforeValidator(_unpack_variable)] = None
    """Site where the facility is located."""
    revision: Annotated[str | None, BeforeValidator(_unpack_variable)] = None
    """Revision of the beamline in case of upgrades."""

    @classmethod
    def from_nexus_entry(
        cls, entry: snx.Group, *, instrument_name: str | None = None
    ) -> Beamline:
        """Construct a Beamline object from a Nexus entry.

        NeXus does not have a standard method for specifying the facility, site, or
        revision. This function only sets those fields for known instruments.

        Parameters
        ----------
        entry:
            ScippNexus group for a NeXus entry.
            The entry needs to contain an ``NXinstrument`` with a 'name' field
            to identify the instrument.
        instrument_name:
            If the entry contains more than one ``NXinstrument`` group, this parameter
            must be the name of one of these groups.

        Returns
        -------
        :
            A Beamline object constructed from the given Nexus entry.
        """
        instrument = _get_unique_nexus_child(entry, snx.NXinstrument, instrument_name)
        instrument_name = _read_optional_nexus_string(instrument, 'name')
        if instrument_name is None:
            raise ValueError("No instrument name found in Nexus entry")

        facility, site = _guess_facility_and_site(instrument_name)

        return cls(
            name=instrument_name,
            facility=facility,
            site=site,
        )


class Measurement(BaseModel):
    """A single measurement.

    Terminology:

        - An "experiment" is the collection of all related measurements,
          typically done during one beamtime.
        - A "measurement" is a single step of data collection.
          It typically corresponds to a single NeXus file or a single entry
          in a NeXus file.

    The ``Measurement`` class represents a single measurement but also includes some
    information about the experiment that this measurement is part of.
    In particular, ``experiment_id`` and ``experiment_doi`` encode
    information about the experiment.
    *All* other fields encode information about a measurement; this includes
    ``start_time`` and ``end_time``.
    """

    title: Annotated[str | None, BeforeValidator(_unpack_variable)]
    """The title of the measurement."""
    run_number: Annotated[str | None, BeforeValidator(_unpack_variable)] = None
    """Run number of the measurement."""
    experiment_id: Annotated[str | None, BeforeValidator(_unpack_variable)] = None
    """An ID for the experiment that this measurement is part of, e.g., proposal ID."""
    experiment_doi: Annotated[str | None, BeforeValidator(_unpack_variable)] = None
    """A DOI for the experiment that this measurement is part of."""
    # These do not support conversion from variables because variables are naive w.r.t.
    # timezones and converting them to datetime.datetime would require extra info.
    start_time: datetime | None = None
    """Date and time when the measurement started."""
    end_time: datetime | None = None
    """Date and time when the measurement ended."""

    @classmethod
    def from_nexus_entry(cls, entry: snx.Group) -> Measurement:
        """Construct a Measurement object from a Nexus entry.

        Parameters
        ----------
        entry:
            ScippNexus group for a NeXus entry.

        Returns
        -------
        :
            An Measurement object constructed from the given Nexus entry.
        """
        return cls(
            title=_read_optional_nexus_string(entry, 'title'),
            run_number=_read_optional_nexus_string(entry, 'entry_identifier'),
            experiment_id=_read_optional_nexus_string(entry, 'experiment_identifier'),
            start_time=_read_optional_nexus_datetime(entry, 'start_time'),
            end_time=_read_optional_nexus_datetime(entry, 'end_time'),
            experiment_doi=None,
        )

    @property
    def run_number_maybe_int(self) -> int | str | None:
        """Return the run number as an int if possible."""
        try:
            return int(self.run_number)
        except ValueError:
            return self.run_number


class Person(BaseModel):
    """A person.

    .. attention::

        Please make sure that you don't specify personal details without permission!

    Specify an ORCID iD whenever possible to make sure the
    person can be uniquely identified.
    The name is always required but is often not unique.
    Other contact information like address or email address are less important
    and can usually be omitted when an ORCID iD is provided.
    """

    name: Annotated[str, BeforeValidator(_unpack_variable)]
    """Free form name of the person."""
    orcid_id: Annotated[ORCIDiD | None, BeforeValidator(_unpack_variable)] = None
    """ORCID iD of the person."""

    corresponding: bool = False
    """Whether the person is the corresponding / contact author."""
    owner: bool = True
    """Whether the person owns the data."""
    role: Annotated[str | None, BeforeValidator(_unpack_variable)] = None
    """The role that the person played in collecting / processing the data.

    `NeXus <https://manual.nexusformat.org/classes/base_classes/NXuser.html#nxuser>`_
    and
    `CIF <https://github.com/COMCIFS/cif_core/blob/6f8502e81b623eb0fd779c79efaf191d49fa198c/cif_core.dic#L15167>`_
    list possible roles.
    """

    address: Annotated[str | None, BeforeValidator(_unpack_variable)] = None
    """Physical (work) address of the person."""
    email: Annotated[EmailStr | None, BeforeValidator(_unpack_variable)] = None
    """Email address of the person."""
    affiliation: Annotated[str | None, BeforeValidator(_unpack_variable)] = None
    """Affiliation of the person."""


class Software(BaseModel):
    """A piece of software.

    The piece of software should be specified as precisely as possible.
    For example, a release version of ScippNeutron could be specified as follows:

    >>> from scippneutron.metadata import Software
    >>> software = Software(
    ...     name='ScippNeutron',
    ...     version='24.11.0',
    ...     url='https://github.com/scipp/scippneutron/releases/tag/24.11.0',
    ...     doi='https://doi.org/10.5281/zenodo.14139599',
    ... )

    A development version might include a Git hash in the version.
    Alternative methods can be used for projects that do not use Git.
    But the software should be specified as precisely as possible.
    For example:

    >>> software = Software(
    ...     name='ScippNeutron',
    ...     version='24.11.1.dev8+g10d09ab0',
    ...     url='https://github.com/scipp/scippneutron',
    ... )

    The URL can either point to the source code, a release artifact, or a package
    index, such as ``pypi.org`` or ``anaconda.org``.
    """

    name: str
    """Name of the piece of software."""
    version: str
    """Complete version of the piece of software."""
    url: str | None = None
    """URL to the concrete version of the software.

    If no URL for a concrete version is available,
    a URL of the project or source code may be used.
    """
    doi: str | None = None
    """DOI of the concrete version of the software.

    If there is no DOI for the concrete version,
    a general DOI for the software may be used.
    """

    @classmethod
    def from_package_metadata(cls, package_name: str) -> Software:
        """Construct a Software instance from the metadata of an installed package.

        This function attempts to deduce all information it can from package metadata.
        But it only has access to the information that is encoded in the package.
        It therefore returns the base project URL instead of a concrete release URL,
        and it does not return a DOI.

        Parameters
        ----------
        package_name:
            The name of the Python package.

        Returns
        -------
        :
            A Software instance.
        """
        return cls(
            name=package_name,
            version=_deduce_package_version(package_name),
            url=_deduce_package_source_url(package_name),
            doi=None,
        )

    @property
    def name_version(self) -> str:
        """The name and version of the software, separated by a space."""
        return f'{self.name} {self.version}'

    @property
    def compact_repr(self) -> str:
        """A representation of this software as a single short string."""
        if self.url:
            return f'{self.name_version} ({self.url})'
        return self.name_version


def _deduce_package_version(package_name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package_name)
    except PackageNotFoundError:
        # Either the package is not installed or has no metadata.
        from importlib import import_module

        try:
            package = import_module(package_name)
        except ModuleNotFoundError as e:
            raise e from None

        try:
            return package.__version__
        except AttributeError:
            raise RuntimeError(
                f"Package '{package_name}' has no metadata and no "
                f"__version__ attribute. Specify the version manually."
            ) from None


def _deduce_package_source_url(package_name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, metadata

    try:
        meta = metadata(package_name)
    except PackageNotFoundError:
        # Either the package is not installed or has no metadata.
        return None

    if not (urls := meta.get_all("project-url")):
        return None

    try:
        return next(
            url.split(',')[-1].strip() for url in urls if url.startswith("Source")
        )
    except StopIteration:
        return None


class SourceType(enum.Enum):
    """Type of source.

    Names are based on NeXus definitions.
    """

    SpallationNeutronSource = 'Spallation Neutron Source'
    """A spallation neutron source."""
    ReactorNeutronSource = 'Reactor Neutron Source'
    """A reactor neutron source."""
    SynchrotronXraySource = 'Synchrotron X-ray Source'
    """A synchrotron X-ray source."""


class RadiationProbe(enum.Enum):
    """Type of radiation probe.

    Names are based on NeXus definitions.
    """

    Neutron = 'neutron'
    """Neutron probe."""
    Xray = 'X-ray'
    """X-ray probe."""


class Source(BaseModel):
    """Information about a neutron source.

    The ESS source is provided as ``scippneutron.metadata.ESS_SOURCE``.
    """

    name: Annotated[str | None, BeforeValidator(_unpack_variable)] = None
    """Name of the source."""
    source_type: SourceType
    """Type of this source."""
    probe: RadiationProbe
    """Radiation probe of the source."""


ESS_SOURCE = Source(
    name="ESS Butterfly",
    source_type=SourceType.SpallationNeutronSource,
    probe=RadiationProbe.Neutron,
)
ESS_SOURCE.__doc__ = """Default parameters of the ESS source."""


def _read_optional_nexus_string(group: snx.Group | None, key: str) -> str | None:
    if group is None:
        return None
    if (ds := group.get(key)) is not None:
        return ds[()]
    return None


def _read_optional_nexus_datetime(group: snx.Group | None, key: str) -> datetime | None:
    if (s := _read_optional_nexus_string(group, key)) is not None:
        return parse_datetime(s)
    return None


def _get_unique_nexus_child(
    entry: snx.Group, nx_class: type, name: str | None
) -> snx.Group | None:
    if name is not None:
        return entry.get(name)
    children = entry[nx_class]
    if len(children) > 1:
        raise RuntimeError(
            f"Got multiple {nx_class.__name__} in NeXus entry '{entry.name}'"
        )
    if len(children) == 0:
        return None
    return next(iter(children.values()))


# More instruments may be added as needed.
# All instrument names are lowercase.
_FACILITY_PER_INSTRUMENT: dict[str, str | tuple[str, str]] = {
    # ESS
    'beer': 'ESS',
    'bifrost': 'ESS',
    'cspec': 'ESS',
    'dream': 'ESS',
    'estia': 'ESS',
    'freia': 'ESS',
    'heimdal': 'ESS',
    'loki': 'ESS',
    'magic': 'ESS',
    'miracles': 'ESS',
    'nmx': 'ESS',
    'odin': 'ESS',
    'skadi': 'ESS',
    'tbl': 'ESS',
    'trex': 'ESS',
    'vespa': 'ESS',
    # SINQ
    'amor': ('SINQ', 'PSI'),
}


# NeXus provides no way to specify the facility.
# But we can usually guess it based on the instrument name.
def _guess_facility_and_site(
    instrument_name: str | None,
) -> tuple[str | None, str | None]:
    if instrument_name is None:
        return None, None

    match _FACILITY_PER_INSTRUMENT.get(instrument_name.lower()):
        case None:
            return None, None
        case (facility, site):
            return facility, site
        case facility:
            return facility, facility
