# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import itertools
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class ORCIDiD:
    """An ORCID iD.

    Ensures that the id is valid during initialization.
    See https://support.orcid.org/hc/en-us/articles/360006897674-Structure-of-the-ORCID-Identifier
    This class can be used with Pydantic models.

    Examples
    --------

        >>> from scippneutron.metadata import ORCIDiD
        >>> orcid_id = ORCIDiD('0000-0000-0000-0001')
        >>> str(orcid_id)
        'https://orcid.org/0000-0000-0000-0001'

    Or equivalently with an explicit resolver:

        >>> orcid_id = ORCIDiD('https://orcid.org/0000-0000-0000-0001')
        >>> str(orcid_id)
        'https://orcid.org/0000-0000-0000-0001'
    """

    __slots__ = ('_orcid_id',)

    def __init__(self, orcid_id: str | ORCIDiD) -> None:
        if isinstance(orcid_id, ORCIDiD):
            self._orcid_id: str = orcid_id._orcid_id
        else:
            self._orcid_id = _parse_id(orcid_id)

    def __str__(self) -> str:
        return self._orcid_id

    def __repr__(self) -> str:
        return f'ORCIDiD({self!s})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ORCIDiD):
            return self._orcid_id == other._orcid_id
        if isinstance(other, str):
            try:
                return self._orcid_id == _parse_id(other)
            except ValueError:  # other is not a valid ORCID iD
                return False
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        if (b := self.__eq__(other)) is NotImplemented:
            return NotImplemented
        return not b

    def __hash__(self) -> int:
        return hash(self._orcid_id)

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


_ORCID_RESOLVER: str = 'https://orcid.org'


def _parse_id(value: str) -> str:
    parts = value.rsplit('/', 1)
    if len(parts) == 2:
        resolver, orcid_id = parts
        if resolver != _ORCID_RESOLVER:
            # Must be the correct ORCID URL.
            raise ValueError(
                f"Invalid ORCID URL: '{resolver}'. Must be '{_ORCID_RESOLVER}'"
            )
    else:
        value = f'{_ORCID_RESOLVER}/{value}'
        (orcid_id,) = parts
    _check_id(orcid_id)
    return value


def _check_id(orcid_id: str) -> None:
    segments = orcid_id.split('-')
    if len(segments) != 4 or not all(len(s) == 4 for s in segments):
        # Must have 4 blocks of 4 digits each.
        raise ValueError(f"Invalid ORCID iD: '{orcid_id}'. Incorrect structure.")
    if _orcid_id_checksum(segments) != orcid_id[-1]:
        # Checksum must match the last digit.
        raise ValueError(f"Invalid ORCID iD: '{orcid_id}'. Checksum does not match.")


def _orcid_id_checksum(segments: list[str]) -> str:
    total = 0
    for d in map(int, itertools.islice(itertools.chain(*segments), 4 * 4 - 1)):
        total = (total + d) * 2
    result = (12 - total % 11) % 11
    return 'X' if result == 10 else str(result)
