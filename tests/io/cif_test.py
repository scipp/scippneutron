# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import io

from scippneutron.io import cif


def write_to_str(block: cif.Block) -> str:
    buffer = io.StringIO()
    block.write(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def test_write_block_empty():
    block = cif.Block('a block-name')
    res = write_to_str(block)
    assert res == 'data_a block-name\n'


def test_write_block_single_pair():
    block = cif.Block('single', [{'audit.creation_method': 'written_by_scippneutron'}])
    res = write_to_str(block)
    assert (
        res
        == '''data_single

_audit.creation_method written_by_scippneutron
'''
    )


def test_write_block_single_pair_space():
    block = cif.Block('single', [{'audit.creation_method': 'written by scippneutron'}])
    res = write_to_str(block)
    assert (
        res
        == '''data_single

_audit.creation_method 'written by scippneutron'
'''
    )


def test_write_block_single_pair_single_quote():
    block = cif.Block(
        'single', [{'audit.creation_method': "written by 'scippneutron'"}]
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_single

_audit.creation_method "written by 'scippneutron'"
'''
    )


def test_write_block_single_pair_double_quote():
    block = cif.Block(
        'single', [{'audit.creation_method': 'written by "scippneutron"'}]
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_single

_audit.creation_method 'written by "scippneutron"'
'''
    )


def test_write_block_single_pair_both_quotes():
    block = cif.Block(
        'single', [{'audit.creation_method': """'written by "scippneutron"'"""}]
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_single

_audit.creation_method
; 'written by "scippneutron"'
;
'''
    )


def test_write_block_single_pair_newline():
    block = cif.Block(
        'single',
        [{'audit.creation_method': "written by scippneutron\n    version 2000"}],
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_single

_audit.creation_method
; written by scippneutron
    version 2000
;
'''
    )


def test_write_block_single_pair_single_line_comment():
    block = cif.Block('comment')
    block.add({'diffrn_radiation.probe': 'neutron'}, comment='a comment')
    res = write_to_str(block)
    assert (
        res
        == '''data_comment

# a comment
_diffrn_radiation.probe neutron
'''
    )


def test_write_block_single_pair_multi_line_comment():
    block = cif.Block('comment')
    block.add(
        {'diffrn_radiation.probe': 'neutron'},
        comment='Guessing that\nthis is the\n  correct probe',
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_comment

# Guessing that
# this is the
#   correct probe
_diffrn_radiation.probe neutron
'''
    )


def test_write_block_multiple_pairs():
    block = cif.Block(
        'multiple',
        [
            {
                'audit.creation_method': 'written_by_scippneutron',
                'audit.creation_date': '2023-12-01T13:52:00Z',
            }
        ],
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_multiple

_audit.creation_method written_by_scippneutron
_audit.creation_date 2023-12-01T13:52:00Z
'''
    )


def test_write_block_multiple_chunks():
    block = cif.Block(
        'multiple',
        [
            {
                'audit.creation_method': 'written_by_scippneutron',
                'audit.creation_date': '2023-12-01T13:52:00Z',
            }
        ],
    )
    block.add({'diffrn_radiation.probe': 'neutron'})
    res = write_to_str(block)
    assert (
        res
        == '''data_multiple

_audit.creation_method written_by_scippneutron
_audit.creation_date 2023-12-01T13:52:00Z

_diffrn_radiation.probe neutron
'''
    )


def test_write_block_multiple_chunks_comment():
    block = cif.Block(
        'multiple',
        [
            {
                'audit.creation_method': 'written_by_scippneutron',
                'audit.creation_date': '2023-12-01T13:52:00Z',
            }
        ],
    )
    block.add({'diffrn_radiation.probe': 'neutron'}, comment='Guessed')
    res = write_to_str(block)
    assert (
        res
        == '''data_multiple

_audit.creation_method written_by_scippneutron
_audit.creation_date 2023-12-01T13:52:00Z

# Guessed
_diffrn_radiation.probe neutron
'''
    )
