from __future__ import annotations

from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

_ORCID_PREFIX: str = 'https://orcid.org'


class ORCIDiD:
    """An ORCID iD.

    Ensures that the id is valid during initialization.
    See https://support.orcid.org/hc/en-us/articles/360006897674-Structure-of-the-ORCID-Identifier
    This class can be used with Pydantic models.

    Examples
    --------

        >>> from scippneutron.meta import ORCIDiD
        >>> orcid_id = ORCIDiD('0000-0000-0000-0001')
        >>> orcid_id
        https://orcid.org/0000-0000-0000-0001

    Or equivalently with an explicit prefix:

        >>> orcid_id = ORCIDiD('https://orcid.org/0000-0000-0000-0001')
        >>> orcid_id
        https://orcid.org/0000-0000-0000-0001
    """

    __slots__ = ('_orcid_id',)

    def __init__(self, orcid_id: str | ORCIDiD) -> None:
        if isinstance(orcid_id, ORCIDiD):
            self._orcid_id: str = orcid_id._orcid_id
        else:
            self._orcid_id = _parse_id(orcid_id)

    def __str__(self) -> str:
        return f'{_ORCID_PREFIX}/{self._orcid_id}'

    def __repr__(self) -> str:
        return f'ORCIDiD({self!s})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ORCIDiD):
            return self._orcid_id == other._orcid_id
        if isinstance(other, str):
            return self._orcid_id == _parse_id(other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(str(self._orcid_id))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            _parse_pydantic,
            core_schema.union_schema(
                [core_schema.is_instance_schema(ORCIDiD), core_schema.str_schema()]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls.__str__, info_arg=False, return_schema=core_schema.str_schema()
            ),
        )


def _parse_pydantic(value: str | ORCIDiD) -> ORCIDiD:
    return ORCIDiD(value)


def _parse_id(value: str) -> str:
    parts = value.rsplit('/', 1)
    if len(parts) == 2:
        prefix, orcid_id = parts
        if prefix != _ORCID_PREFIX:
            # must be the correct ORCID URL
            raise ValueError(
                f"Invalid ORCID URL: '{prefix}'. Must be '{_ORCID_PREFIX}'"
            )
    else:
        (orcid_id,) = parts

    segments = orcid_id.split('-')
    if len(segments) != 4 or not all(len(s) == 4 for s in segments):
        # must have four blocks of numbers
        # and each block must have 4 digits
        raise ValueError(f"Invalid ORCID iD: '{orcid_id}'. Incorrect structure.")
    if _orcid_id_checksum(orcid_id) != orcid_id[-1]:
        # checksum must match the last digit
        raise ValueError(f"Invalid ORCID iD: '{orcid_id}'. Checksum does not match.")

    return orcid_id


def _orcid_id_checksum(orcid_id: str) -> str:
    total = 0
    for c in orcid_id[:-1].replace('-', ''):
        total = (total + int(c)) * 2
    result = (12 - total % 11) % 11
    return 'X' if result == 10 else str(result)
