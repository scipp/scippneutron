# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
from pydantic import BaseModel

from scippneutron.metadata import ORCIDiD


def test_orcid_init_no_resolver() -> None:
    orcid_id = ORCIDiD('0000-0000-0000-0001')
    assert str(orcid_id) == 'https://orcid.org/0000-0000-0000-0001'
    assert repr(orcid_id) == 'ORCIDiD(https://orcid.org/0000-0000-0000-0001)'


def test_orcid_init_valid_resolver() -> None:
    orcid_id = ORCIDiD('https://orcid.org/0000-0000-0000-0001')
    assert str(orcid_id) == 'https://orcid.org/0000-0000-0000-0001'
    assert repr(orcid_id) == 'ORCIDiD(https://orcid.org/0000-0000-0000-0001)'


def test_orcid_init_invalid_resolver() -> None:
    with pytest.raises(ValueError, match='Invalid ORCID URL'):
        ORCIDiD('https://my-orcid.org/0000-0000-0000-0001')


def test_orcid_init_invalid_structure() -> None:
    with pytest.raises(ValueError, match='Incorrect structure'):
        ORCIDiD('0000-0000-0001')
    with pytest.raises(ValueError, match='Incorrect structure'):
        ORCIDiD('0000-0000-0001-123')


def test_orcid_init_invalid_checksum() -> None:
    with pytest.raises(ValueError, match='Checksum does not match'):
        ORCIDiD('0000-0000-0000-0000')


def test_orcid_eq() -> None:
    full = ORCIDiD('https://orcid.org/0000-0000-0001-0007')
    no_resolver = ORCIDiD('0000-0000-0001-0007')
    full_str = 'https://orcid.org/0000-0000-0001-0007'
    no_resolver_str = '0000-0000-0001-0007'

    assert full == full
    assert full == no_resolver
    assert full == full_str
    assert full == no_resolver_str

    assert no_resolver == full
    assert no_resolver == no_resolver
    assert no_resolver == full_str
    assert no_resolver == no_resolver_str

    assert full_str == full
    assert full_str == no_resolver
    assert no_resolver_str == full
    assert no_resolver_str == no_resolver


def test_orcid_ne() -> None:
    ref = ORCIDiD('https://orcid.org/0000-0000-0000-0001')
    ref_str = 'https://orcid.org/0000-0000-0000-0001'

    assert ORCIDiD('https://orcid.org/0000-0000-0001-0007') != ref
    assert ORCIDiD('0000-0000-0001-0007') != ref
    assert ORCIDiD('https://orcid.org/0000-0000-0001-0007') != ref_str
    assert ORCIDiD('0000-0000-0001-0007') != ref_str

    assert ORCIDiD('https://orcid.org/0000-0000-0001-0007') != "1-007"
    assert ORCIDiD('0000-0000-0001-0007') != "1-007"
    assert not (ORCIDiD('https://orcid.org/0000-0000-0001-0007') == "1-007")
    assert not (ORCIDiD('0000-0000-0001-0007') == "1-007")

    assert ORCIDiD('https://orcid.org/0000-0000-0001-0007') != 1007
    assert ORCIDiD('0000-0000-0001-0007') != 1007
    assert not (ORCIDiD('https://orcid.org/0000-0000-0001-0007') == 1007)
    assert not (ORCIDiD('0000-0000-0001-0007') == 1007)


def test_orcid_hash_is_deterministic() -> None:
    a = ORCIDiD('https://orcid.org/0000-0000-0000-0001')
    b = ORCIDiD('https://orcid.org/0000-0000-0001-0007')
    assert hash(a) == hash(a)
    assert hash(b) == hash(b)


def test_pydantic_model_from_orcid_id() -> None:
    class Model(BaseModel):
        orcid_id: ORCIDiD

    m = Model(orcid_id=ORCIDiD('0000-0000-0000-0001'))
    assert m.orcid_id == ORCIDiD('0000-0000-0000-0001')


def test_pydantic_model_from_str() -> None:
    class Model(BaseModel):
        orcid_id: ORCIDiD

    m = Model(orcid_id='0000-0000-0000-0001')  # type: ignore[arg-type]
    assert m.orcid_id == ORCIDiD('0000-0000-0000-0001')


def test_pydantic_model_serialize() -> None:
    class Model(BaseModel):
        orcid_id: ORCIDiD

    m = Model(orcid_id=ORCIDiD('0000-0000-0000-0001'))
    res = m.model_dump()
    assert res == {'orcid_id': 'https://orcid.org/0000-0000-0000-0001'}
