# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import io

import pytest
import scipp as sc

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


def test_write_block_single_pair_string():
    block = cif.Block('single', [{'audit.creation_method': 'written_by_scippneutron'}])
    res = write_to_str(block)
    assert (
        res
        == '''data_single

_audit.creation_method written_by_scippneutron
'''
    )


def test_write_block_single_pair_string_variable():
    block = cif.Block(
        'single', [{'audit.creation_method': sc.scalar('written_by_scippneutron')}]
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_single

_audit.creation_method written_by_scippneutron
'''
    )


def test_write_block_single_pair_number():
    block = cif.Block('number', [{'cell.angle_alpha': 62}])
    res = write_to_str(block)
    assert (
        res
        == '''data_number

_cell.angle_alpha 62
'''
    )


@pytest.mark.parametrize('unit', (None, 'deg'))
def test_write_block_single_pair_number_variable(unit):
    block = cif.Block('number', [{'cell.angle_alpha': sc.scalar(93, unit=unit)}])
    res = write_to_str(block)
    assert (
        res
        == '''data_number

_cell.angle_alpha 93
'''
    )


@pytest.mark.parametrize('unit', (None, 'deg'))
def test_write_block_single_pair_number_error(unit):
    block = cif.Block(
        'number', [{'cell.angle_alpha': sc.scalar(93.2, variance=2.1**2, unit=unit)}]
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_number

_cell.angle_alpha 93(2)
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


def test_write_block_single_loop_one_column():
    env = sc.array(dims=['x'], values=['water', 'sulfur'])
    block = cif.Block('looped', [cif.Loop({'diffrn.ambient_environment': env})])
    res = write_to_str(block)
    assert (
        res
        == '''data_looped

loop_
_diffrn.ambient_environment
water
sulfur
'''
    )


def test_write_block_single_loop_two_columns():
    env = sc.array(dims=['x'], values=['water', 'sulfur'])
    id_ = sc.array(dims=['x'], values=['123', 'x6a'])
    block = cif.Block(
        'looped', [cif.Loop({'diffrn.ambient_environment': env, 'diffrn.id': id_})]
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_looped

loop_
_diffrn.ambient_environment
_diffrn.id
water 123
sulfur x6a
'''
    )


def test_write_block_single_loop_multi_line_string():
    env = sc.array(dims=['x'], values=['water\nand some salt', 'sulfur'])
    id_ = sc.array(dims=['x'], values=['123', 'x6a'])
    block = cif.Block(
        'looped', [cif.Loop({'diffrn.ambient_environment': env, 'diffrn.id': id_})]
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_looped

loop_
_diffrn.ambient_environment
_diffrn.id
; water
and some salt
;
123
sulfur
x6a
'''
    )


def test_write_block_single_loop_numbers():
    coeff = sc.array(dims=['cal'], values=[3.65, -0.012, 1.2e-5])
    power = sc.array(dims=['cal'], values=[0, 1, 2])
    id_ = sc.array(dims=['cal'], values=['tzero', 'DIFC', 'DIFA'])
    block = cif.Block(
        'looped',
        [
            cif.Loop(
                {
                    'pd_calib_d_to_tof.id': id_,
                    'pd_calib_d_to_tof.power': power,
                    'pd_calib_d_to_tof.coeff': coeff,
                }
            )
        ],
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_looped

loop_
_pd_calib_d_to_tof.id
_pd_calib_d_to_tof.power
_pd_calib_d_to_tof.coeff
tzero 0 3.65
DIFC 1 -0.012
DIFA 2 1.2e-05
'''
    )


def test_write_block_single_loop_numbers_errors():
    coeff = sc.array(
        dims=['cal'],
        values=[3.65, -0.012, 1.2e-5],
        variances=[0.13**2, 0.001**2, 2e-6**2],
    )
    power = sc.array(dims=['cal'], values=[0, 1, 2])
    id_ = sc.array(dims=['cal'], values=['tzero', 'DIFC', 'DIFA'])
    block = cif.Block(
        'looped',
        [
            cif.Loop(
                {
                    'pd_calib_d_to_tof.id': id_,
                    'pd_calib_d_to_tof.power': power,
                    'pd_calib_d_to_tof.coeff': coeff,
                }
            )
        ],
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_looped

loop_
_pd_calib_d_to_tof.id
_pd_calib_d_to_tof.power
_pd_calib_d_to_tof.coeff
tzero 0 3.65(13)
DIFC 1 -0.0120(10)
DIFA 2 0.000012(2)
'''
    )


def test_write_block_pairs_then_loop():
    env = sc.array(dims=['x'], values=['water', 'sulfur'])
    block = cif.Block(
        'looped',
        [
            {
                'audit.creation_method': 'written by scippneutron',
                'audit.creation_date': '2023-12-01T13:52:00Z',
            },
            cif.Loop({'diffrn.ambient_environment': env}),
        ],
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_looped

_audit.creation_method 'written by scippneutron'
_audit.creation_date 2023-12-01T13:52:00Z

loop_
_diffrn.ambient_environment
water
sulfur
'''
    )


def test_write_block_loop_then_pairs():
    env = sc.array(dims=['x'], values=['water', 'sulfur'])
    block = cif.Block(
        'looped',
        [
            cif.Loop({'diffrn.ambient_environment': env}),
            {
                'audit.creation_method': 'written by scippneutron',
                'audit.creation_date': '2023-12-01T13:52:00Z',
            },
        ],
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_looped

loop_
_diffrn.ambient_environment
water
sulfur

_audit.creation_method 'written by scippneutron'
_audit.creation_date 2023-12-01T13:52:00Z
'''
    )


def test_write_block_pair_then_loop_then_pairs():
    env = sc.array(dims=['x'], values=['water', 'sulfur'])
    block = cif.Block(
        'looped',
        [
            {'diffrn_radiation.probe': 'neutron'},
            cif.Loop({'diffrn.ambient_environment': env}),
            {
                'audit.creation_method': 'written by scippneutron',
                'audit.creation_date': '2023-12-01T13:52:00Z',
            },
        ],
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_looped

_diffrn_radiation.probe neutron

loop_
_diffrn.ambient_environment
water
sulfur

_audit.creation_method 'written by scippneutron'
_audit.creation_date 2023-12-01T13:52:00Z
'''
    )


def test_write_block_two_loops():
    env = sc.array(dims=['env'], values=['water', 'sulfur'])
    author = sc.array(dims=['author'], values=['Ridcully, M.', 'Librarian'])
    email = sc.array(dims=['email'], values=['m.ridcully@uu.am', 'lib@uu.am'])
    block = cif.Block(
        'looped',
        [
            cif.Loop({'diffrn.ambient_environment': env}),
            cif.Loop({'audit_author.name': author, 'audit_author.email': email}),
        ],
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_looped

loop_
_diffrn.ambient_environment
water
sulfur

loop_
_audit_author.name
_audit_author.email
'Ridcully, M.' m.ridcully@uu.am
Librarian lib@uu.am
'''
    )
