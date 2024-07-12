import pytest
from pydantic import BaseModel

from scippneutron.meta import ORCIDiD


def test_orcid_init_no_prefix() -> None:
    orcid_id = ORCIDiD('0000-0000-0000-0001')
    assert str(orcid_id) == 'https://orcid.org/0000-0000-0000-0001'
    assert repr(orcid_id) == 'ORCIDiD(https://orcid.org/0000-0000-0000-0001)'


def test_orcid_init_valid_prefix() -> None:
    orcid_id = ORCIDiD('https://orcid.org/0000-0000-0000-0001')
    assert str(orcid_id) == 'https://orcid.org/0000-0000-0000-0001'
    assert repr(orcid_id) == 'ORCIDiD(https://orcid.org/0000-0000-0000-0001)'


def test_orcid_init_invalid_prefix() -> None:
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
