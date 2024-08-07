from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel, EmailStr

from ._orcid import ORCIDiD

_T = TypeVar('_T', bound=BaseModel)


class MIRGroup(BaseModel):
    _parsers: ClassVar[dict[type, Callable[..., MIRGroup]]] = {}

    @classmethod
    def parser(cls: type[_T], source: type) -> Callable[[Callable[..., _T]], None]:
        def parser_decorator(p: Callable[..., Any]) -> None:
            if source in cls._parsers:
                raise RuntimeError(f"Duplicate parser for {source}")
            cls._parsers[source] = p

        return parser_decorator

    @classmethod
    def parse(cls: type[_T], source_id: Any, *args, **kwargs) -> _T:  # noqa: PYI019
        return cls._parsers[type(source_id)](source_id, *args, **kwargs)


class Person(MIRGroup):
    name: str  # free form even though some file formats require a specific format
    orcid: ORCIDiD | None = None

    corresponding: bool = False  # aka 'contact'; formatted differently in some formats
    owner: bool = True
    role: str | None = None  # CIF has a fixed set of allowed roles

    address: str | None = None
    email: EmailStr | None = None
    affiliation: str | None = None


class Beamline(MIRGroup):
    name: str
    facility: str | None = None
    site: str | None = None
    revision: str | None = None


class Software(MIRGroup):
    name: str
    version: str
    url: str | None = None
