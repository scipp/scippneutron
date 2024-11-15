from __future__ import annotations

from pydantic import BaseModel, EmailStr

from ._orcid import ORCIDiD


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

    name: str
    """Free form name of the person."""
    orcid: ORCIDiD | None = None
    """ORCID iD of the person."""

    corresponding: bool = False
    """Whether the person is the corresponding / contact author."""
    owner: bool = True
    """Whether the person owns the data."""
    role: str | None = None
    """The role that the person played in collecting / processing the data.

    `NeXus <https://manual.nexusformat.org/classes/base_classes/NXuser.html#nxuser>`_
    and
    `CIF <https://github.com/COMCIFS/cif_core/blob/6f8502e81b623eb0fd779c79efaf191d49fa198c/cif_core.dic#L15167>`_
    list possible roles.
    """

    address: str | None = None
    """Physical (work) address of the person."""
    email: EmailStr | None = None
    """Email address of the person."""
    affiliation: str | None = None
    """Affiliation of the person."""


class Beamline(BaseModel):
    """A beamline / instrument.

    ``name`` should be the canonical spelling of the beamline name.
    The location of the beamline is split into ``facility`` and ``site``, where
    a 'facility' is located at a 'site'. For example:

    >>> Beamline(
    ...     name='Amor',
    ...     facility='SINQ',
    ...     site='PSI',
    ... )

    If there is no separate facility and site, omit ``site``:

    >>> Beamline(
    ...     name='ESTIA',
    ...     facility='ESS',
    ...     site=None,  # can be omitted
    ... )

    If the beamline has been upgraded, provide a revision to indicate
    which version of the beamline was used.
    """

    name: str
    """Name of the beamline."""
    facility: str | None = None
    """Facility where the beamline is located."""
    site: str | None = None
    """Site where the facility is located."""
    revision: str | None = None
    """Revision of the beamline in case of upgrades."""


class Software(BaseModel):
    """A piece of software.

    The piece of software should be specified as precisely as possible.
    For example, a release version of ScippNeutron could be specified as follows:

    >>> Software(
    ...     name='ScippNeutron',
    ...     version='24.11.0',
    ...     url='https://github.com/scipp/scippneutron/releases/tag/24.11.0',
    ...     doi='https://doi.org/10.5281/zenodo.14139599',
    ... )

    A development version might include a Git hash in the version.
    Alternative methods can be used for projects that do not use Git.
    But the software should be specified as precisely as possible.
    For example:

    >>> Software(
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
    """URL to the concrete version of the software."""
    doi: str | None = None
    """DOI of the concrete version of the software."""
