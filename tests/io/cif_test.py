# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import io
from datetime import datetime, timezone
from pathlib import Path

import pytest
import scipp as sc

from scippneutron.io import cif


def write_to_str(block: cif.Block) -> str:
    buffer = io.StringIO()
    block.write(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def test_write_block_empty():
    block = cif.Block('a-block-name')
    res = write_to_str(block)
    assert res == 'data_a-block-name\n\n'


def test_write_block_name_with_space():
    with pytest.raises(ValueError, match='Block name must not contain spaces'):
        cif.Block('a block-name with space')


def test_write_block_comment():
    block = cif.Block('a-block-name', comment='some comment\n to describe the block')
    res = write_to_str(block)
    assert (
        res
        == '''# some comment
#  to describe the block
data_a-block-name

'''
    )


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


@pytest.mark.parametrize('unit', [None, 'deg'])
def test_write_block_single_pair_number_variable(unit):
    block = cif.Block('number', [{'cell.angle_alpha': sc.scalar(93, unit=unit)}])
    res = write_to_str(block)
    assert (
        res
        == '''data_number

_cell.angle_alpha 93
'''
    )


@pytest.mark.parametrize('unit', [None, 'deg'])
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


def test_write_block_single_pair_datetime():
    dt = datetime(
        year=2023, month=12, day=1, hour=15, minute=9, second=45, tzinfo=timezone.utc
    )
    block = cif.Block(
        'datetime',
        [
            {
                'audit.creation_date': dt,
            }
        ],
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_datetime

_audit.creation_date 2023-12-01T15:09:45+00:00
'''
    )


def test_write_block_single_pair_datetime_variable():
    block = cif.Block(
        'datetime',
        [
            {
                'audit.creation_date': sc.datetime('2023-12-01T15:12:33'),
            }
        ],
    )
    res = write_to_str(block)
    # No timezone info in the output!
    assert (
        res
        == '''data_datetime

_audit.creation_date 2023-12-01T15:12:33
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


def test_write_block_single_pair_utf8():
    block = cif.Block('utf-8', [{'audit.creation_method': 'Unicode: \xb5\xc5'}])
    res = write_to_str(block)
    assert (
        res
        == r'''data_utf-8

_audit.creation_method 'Unicode: \xb5\xc5'
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


def test_write_block_single_pair_single_line_comment_utf8():
    block = cif.Block('comment')
    block.add({'diffrn_radiation.probe': 'neutron'}, comment='unicode: \xc5')
    res = write_to_str(block)
    assert (
        res
        == r'''data_comment

# unicode: \xc5
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


def test_write_block_single_loop_one_column_comment():
    env = sc.array(dims=['x'], values=['water', 'sulfur'])
    block = cif.Block(
        'looped',
        [
            cif.Loop(
                {'diffrn.ambient_environment': env},
                comment='This data is completely made up!',
            )
        ],
    )
    res = write_to_str(block)
    assert (
        res
        == '''data_looped

# This data is completely made up!
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
    email = sc.array(dims=['author'], values=['m.ridcully@uu.am', 'lib@uu.am'])
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


def test_write_block_core_schema_from_chunk():
    chunk = cif.Chunk(
        {'audit.creation_method': 'written by scippneutron'}, schema=cif.CORE_SCHEMA
    )
    block = cif.Block('block-with-schema', [chunk])
    res = write_to_str(block)
    assert (
        res
        == '''data_block-with-schema

loop_
_audit_conform.dict_name
_audit_conform.dict_version
_audit_conform.dict_location
coreCIF 3.3.0 https://github.com/COMCIFS/cif_core/blob/fc3d75a298fd7c0c3cde43633f2a8616e826bfd5/cif_core.dic

_audit.creation_method 'written by scippneutron'
'''
    )


def test_write_block_core_schema_from_loop():
    author = sc.array(dims=['author'], values=['Ridcully, M.', 'Librarian'])
    email = sc.array(dims=['author'], values=['m.ridcully@uu.am', 'lib@uu.am'])
    loop = cif.Loop(
        {'audit_author.name': author, 'audit_author.email': email},
        schema=cif.CORE_SCHEMA,
    )
    block = cif.Block('block-with-schema', [loop])
    res = write_to_str(block)
    assert (
        res
        == '''data_block-with-schema

loop_
_audit_conform.dict_name
_audit_conform.dict_version
_audit_conform.dict_location
coreCIF 3.3.0 https://github.com/COMCIFS/cif_core/blob/fc3d75a298fd7c0c3cde43633f2a8616e826bfd5/cif_core.dic

loop_
_audit_author.name
_audit_author.email
'Ridcully, M.' m.ridcully@uu.am
Librarian lib@uu.am
'''
    )


def test_write_block_pd_schema_from_chunk():
    chunk = cif.Chunk(
        {'pd_meas.units_of_intensity': '1/(micro ampere)'}, schema=cif.PD_SCHEMA
    )
    block = cif.Block('block-with-schema', [chunk])
    res = write_to_str(block)
    # The order of schemas is arbitrary, so we cannot easily check the whole string.
    assert 'pdCIF' in res
    assert 'coreCIF' in res


def test_write_block_multi_schema_schema():
    core_chunk = cif.Chunk(
        {'audit.creation_method': 'written by scippneutron'}, schema=cif.CORE_SCHEMA
    )
    pd_chunk = cif.Chunk(
        {'pd_meas.units_of_intensity': '1/(micro ampere)'}, schema=cif.PD_SCHEMA
    )
    block = cif.Block('block-with-schema', [core_chunk, pd_chunk])
    res = write_to_str(block)
    # The order of schemas is arbitrary, so we cannot easily check the whole string.
    assert 'pdCIF' in res
    assert 'coreCIF' in res


def test_save_cif_one_block_buffer():
    block1 = cif.Block(
        'block-1', [{'audit.creation_method': 'written by scippneutron'}]
    )
    buffer = io.StringIO()
    cif.save_cif(buffer, block1)
    buffer.seek(0)
    assert (
        buffer.read()
        == r'''#\#CIF_1.1
data_block-1

_audit.creation_method 'written by scippneutron'
'''
    )


def test_save_cif_two_blocks_buffer():
    env = sc.array(dims=['env'], values=['water', 'sulfur'])
    block1 = cif.Block(
        'block-1', [{'audit.creation_method': 'written by scippneutron'}]
    )
    block2 = cif.Block(
        'block-2',
        [
            {'diffrn_radiation.probe': 'neutron'},
            cif.Loop({'diffrn.ambient_environment': env}),
        ],
    )
    buffer = io.StringIO()
    cif.save_cif(buffer, [block1, block2])
    buffer.seek(0)
    assert (
        buffer.read()
        == r'''#\#CIF_1.1
data_block-1

_audit.creation_method 'written by scippneutron'

data_block-2

_diffrn_radiation.probe neutron

loop_
_diffrn.ambient_environment
water
sulfur
'''
    )


@pytest.mark.parametrize('path_type', [str, Path])
def test_save_cif_one_block_file(tmpdir, path_type):
    path = path_type(Path(tmpdir) / "test_save_cif_one_block.cif")
    block1 = cif.Block(
        'block-1', [{'audit.creation_method': 'written by scippneutron'}]
    )

    cif.save_cif(path, block1)
    with open(path) as f:
        assert (
            f.read()
            == r'''#\#CIF_1.1
data_block-1

_audit.creation_method 'written by scippneutron'
'''
        )


def test_loop_requires_1d():
    with pytest.raises(sc.DimensionError):
        cif.Loop({'fake': sc.zeros(sizes={'x': 4, 'y': 3})})


def test_loop_requires_matching_dims():
    with pytest.raises(sc.DimensionError):
        cif.Loop({'a': sc.zeros(sizes={'x': 4}), 'b': sc.zeros(sizes={'x': 3})})
    with pytest.raises(sc.DimensionError):
        cif.Loop({'a': sc.zeros(sizes={'x': 4}), 'b': sc.zeros(sizes={'y': 4})})


def test_block_with_reduced_powder_data():
    da = sc.DataArray(
        sc.array(
            dims=['tof'],
            values=[13.6, 26.0, 9.7],
            variances=[0.81, 1, 0.36],
        ),
        coords={'tof': sc.array(dims=['tof'], values=[1.2, 1.4, 2.3], unit='us')},
    )

    block = cif.Block('reduced', [])
    block.add_reduced_powder_data(da)
    res = write_to_str(block)

    assert 'pdCIF' in res
    assert 'coreCIF' in res

    _, _, tof_loop = res.split('\n\n')
    assert (
        tof_loop
        == '''loop_
_pd_meas.time_of_flight
_pd_proc.intensity_net
_pd_proc.intensity_net_su
1.2 13.6 0.9
1.4 26.0 1.0
2.3 9.7 0.6
'''
    )


def test_block_with_reduced_powder_data_custom_unit():
    da = sc.DataArray(
        sc.array(dims=['tof'], values=[13.6, 26.0, 9.7], unit='counts'),
        coords={'tof': sc.array(dims=['tof'], values=[1.2, 1.4, 2.3], unit='us')},
    )

    block = cif.Block('reduced', [])
    block.add_reduced_powder_data(da)
    res = write_to_str(block)

    assert 'pdCIF' in res
    assert 'coreCIF' in res

    _, _, tof_loop = res.split('\n\n')
    assert (
        tof_loop
        == '''# Unit of intensity: [counts]
loop_
_pd_meas.time_of_flight
_pd_proc.intensity_net
1.2 13.6
1.4 26.0
2.3 9.7
'''
    )


def test_block_with_reduced_powder_data_bad_dim():
    da = sc.DataArray(
        sc.array(
            dims=['time'],
            values=[13.6, 26.0, 9.7],
        ),
        coords={'time': sc.array(dims=['time'], values=[1.2, 1.4, 2.3], unit='us')},
    )

    block = cif.Block('reduced', [])
    with pytest.raises(sc.CoordError):
        block.add_reduced_powder_data(da)


def test_block_with_reduced_powder_data_bad_name():
    da = sc.DataArray(
        sc.array(
            dims=['tof'],
            values=[13.6, 26.0, 9.7],
        ),
        coords={'tof': sc.array(dims=['tof'], values=[1.2, 1.4, 2.3], unit='us')},
        name='bad',
    )

    block = cif.Block('reduced', [])
    with pytest.raises(
        ValueError, match='Unrecognized name for reduced powder data: bad'
    ):
        block.add_reduced_powder_data(da)


def test_block_with_reduced_powder_data_bad_coord_unit():
    da = sc.DataArray(
        sc.array(dims=['tof'], values=[13.6, 26.0, 9.7]),
        coords={'tof': sc.array(dims=['tof'], values=[1.2, 1.4, 2.3], unit='ns')},
    )

    block = cif.Block('reduced', [])
    with pytest.raises(sc.UnitError):
        block.add_reduced_powder_data(da)


def test_block_powder_calibration():
    da = sc.DataArray(
        sc.array(dims=['cal'], values=[1.2, 4.5, 6.7]),
        coords={'power': sc.array(dims=['cal'], values=[0, 1, -1])},
    )
    block = cif.Block('cal', [])
    block.add_powder_calibration(da)
    res = write_to_str(block)

    assert 'pdCIF' in res
    assert 'coreCIF' in res

    _, _, cal_loop = res.split('\n\n')
    assert (
        cal_loop
        == '''loop_
_pd_calib_d_to_tof.id
_pd_calib_d_to_tof.power
_pd_calib_d_to_tof.coeff
tzero 0 1.2
DIFC 1 4.5
DIFB -1 6.7
'''
    )
