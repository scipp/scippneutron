from dataclasses import dataclass

import pytest

from scippneutron import meta

# TODO reset parsers


def test_custom_beamline_parser() -> None:
    @dataclass
    class SimpleBeamlineSource:
        name: str
        facility: str

    @meta.Beamline.parser(SimpleBeamlineSource)
    def _(source: SimpleBeamlineSource) -> meta.Beamline:
        return meta.Beamline(name=source.name, facility=source.facility)

    s = SimpleBeamlineSource(name='Xperiment', facility='SuperBeam')
    beamline = meta.Beamline.parse(s)
    expected = meta.Beamline(name='Xperiment', facility='SuperBeam')
    assert beamline == expected


def test_custom_person_parser_with_extra_arg() -> None:
    @meta.Person.parser(dict)
    def _(source: dict[str, str], corresponding: bool) -> meta.Person:
        return meta.Person(name=source['name'], corresponding=corresponding)

    person = meta.Person.parse(
        {'name': 'Jane Doe', 'role': 'creator'}, corresponding=True
    )
    expected = meta.Person(name='Jane Doe', corresponding=True)
    assert person == expected


def test_no_duplicate_parsers_allowed() -> None:
    class Source:
        pass

    @meta.Beamline.parser(Source)
    def _(source: Source) -> meta.Beamline:
        return meta.Beamline(name='a')

    with pytest.raises(RuntimeError, match='Duplicate parser'):

        @meta.Beamline.parser(Source)
        def _(source: Source) -> meta.Beamline:
            return meta.Beamline(name='b')
