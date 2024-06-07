# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import string
from io import StringIO

import numpy as np
import pytest
import scipp as sc
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from packaging.version import Version
from scipp.testing import strategies as scst

import scippneutron as scn


@st.composite
def one_dim_data_arrays(
    draw: st.DrawFn, min_n_coords: int = 1, max_n_coords: int = 5
) -> sc.DataArray:
    data = draw(scst.variables(ndim=1, dtype='float64', with_variances=True))
    # See https://github.com/scipp/scipp/issues/3052
    data.variances = abs(data.variances)
    coords = draw(
        st.dictionaries(
            keys=st.text(),
            values=scst.variables(
                sizes=data.sizes, dtype='float64', with_variances=False
            ),
            min_size=min_n_coords,
            max_size=max_n_coords,
        )
    )
    return sc.DataArray(data, coords=coords)


def headers() -> st.SearchStrategy[str]:
    # Using only ASCII characters and excluding \r because Numpy and/or splitlines
    # do something funny with some other characters.
    return st.text(string.ascii_letters + string.digits + string.punctuation + ' \t\n')


def save_to_buffer(da: sc.DataArray, coord: str | None = None, **kwargs) -> StringIO:
    buffer = StringIO()
    scn.io.save_xye(buffer, da, coord=coord, **kwargs)
    buffer.seek(0)
    return buffer


def roundtrip(da: sc.DataArray, coord: str | None = None, **kwargs) -> sc.DataArray:
    buffer = save_to_buffer(da, coord, **kwargs)
    return scn.io.xye.load_xye(
        buffer,
        dim=da.dim,
        coord=coord,
        unit=da.unit,
        coord_unit=da.coords[coord].unit if coord is not None else None,
    )


@given(initial=one_dim_data_arrays(), header=headers(), data=st.data())
def test_roundtrip(initial, header, data):
    coord_name = data.draw(st.sampled_from(list(initial.coords.keys())))
    loaded = roundtrip(initial, coord=coord_name, header=header)
    assert set(loaded.coords.keys()) == {coord_name}
    # Using allclose instead of identical because the format might lose some precision.
    # Especially in the variances -> stddevs conversion.
    assert sc.allclose(
        loaded.coords[coord_name], initial.coords[coord_name], equal_nan=True
    )
    assert sc.allclose(loaded.data, initial.data, equal_nan=True)


@given(da=one_dim_data_arrays(), data=st.data())
def test_saved_file_contains_data_table(da, data):
    coord_name = data.draw(st.sampled_from(list(da.coords.keys())))
    file_contents = save_to_buffer(da, coord=coord_name).getvalue()
    for i, line in enumerate(
        filter(
            lambda line: line and not line.startswith('#'), file_contents.split('\n')
        )
    ):
        x, y, e = map(float, line.split(' '))
        np.testing.assert_allclose(x, da.coords[coord_name][i].value)
        np.testing.assert_allclose(y, da[i].value)
        np.testing.assert_allclose(e, np.sqrt(da[i].variance))


@given(initial=one_dim_data_arrays(max_n_coords=1))
@settings(max_examples=20)
def test_save_deduce_coord_data_with_one_coord(initial):
    coord_name = next(iter(initial.coords.keys()))
    loaded = roundtrip(initial)
    loaded = loaded.rename_dims({loaded.dim: initial.dim})
    # roundtrip cannot deduce coord name and unit in this case.
    loaded_coord = next(iter(loaded.coords.values()))
    loaded_coord.unit = initial.coords[coord_name].unit
    assert sc.allclose(loaded_coord, initial.coords[coord_name], equal_nan=True)


@given(initial=one_dim_data_arrays(), data=st.data())
@settings(max_examples=20)
def test_save_deduce_coord_dim_coord(initial, data):
    dim = initial.dim
    initial_coord = data.draw(
        scst.variables(sizes=initial.sizes, dtype='float64', with_variances=False)
    )
    initial.coords[dim] = initial_coord.rename({initial_coord.dim: dim})

    loaded = roundtrip(initial)
    # roundtrip cannot deduce coord name and unit in this case.
    # loaded_coord = next(iter(loaded.coords.values()))
    loaded_coord = loaded.coords[dim]
    loaded_coord.unit = initial.coords[dim].unit
    assert sc.allclose(loaded_coord, initial.coords[dim], equal_nan=True)


@given(da=one_dim_data_arrays(min_n_coords=2))
@settings(max_examples=20)
def test_save_cannot_deduce_coord_if_multiple_non_dim_coords(da):
    # It can deduce if there is a dim-coord, exclude that.
    assume(da.dim not in da.coords)
    with pytest.raises(
        ValueError,
        match='Cannot deduce which coordinate to save because the data '
        'has more than one and no dimension-coordinate',
    ):
        save_to_buffer(da)


@given(da=one_dim_data_arrays(max_n_coords=1))
@settings(max_examples=20)
def test_input_must_have_at_least_one_coord(da):
    for c in list(da.coords.keys()):
        del da.coords[c]
    with pytest.raises(
        ValueError, match='Cannot save data to XYE file because it has no coordinates'
    ):
        save_to_buffer(da)


@given(da=one_dim_data_arrays(), data=st.data())
@settings(max_examples=20)
def test_input_must_have_variances(da, data):
    da.variances = None
    coord_name = data.draw(st.sampled_from(list(da.coords.keys())))
    with pytest.raises(sc.VariancesError):
        save_to_buffer(da, coord=coord_name)


@given(da=one_dim_data_arrays(), data=st.data())
@settings(max_examples=20)
def test_cannot_save_data_with_bin_edges(da, data):
    coord_name = data.draw(st.sampled_from(list(da.coords.keys())))
    da.coords[coord_name] = sc.concat(
        [da.coords[coord_name], sc.scalar(0.0, unit=da.coords[coord_name].unit)],
        dim=da.dim,
    )
    with pytest.raises(sc.CoordError):
        save_to_buffer(da, coord=coord_name)


@given(da=one_dim_data_arrays(), data=st.data())
@settings(max_examples=20)
def test_cannot_save_data_with_masks(da, data):
    mask = data.draw(scst.variables(sizes=da.sizes, dtype=bool))
    da.masks[data.draw(st.text())] = mask
    with pytest.raises(
        ValueError, match='Cannot save data to XYE file because it has masks'
    ):
        save_to_buffer(da, coord=next(iter(da.coords)))


@pytest.mark.skipif(
    Version(sc.__version__) < Version('24.1'),
    reason='This use of the dataarrays strategy needs Scipp >= 24.*',
)
@given(data=st.data())
@settings(max_examples=20)
def test_input_must_be_one_dimensional(data):
    da = data.draw(
        scst.dataarrays(
            data_args={
                'ndim': st.integers(min_value=2, max_value=4),
                'dtype': 'float64',
                'with_variances': True,
            },
            masks=False,
            bin_edges=False,
        )
    )
    assume(da.coords)
    coord_name = data.draw(st.sampled_from(list(da.coords.keys())))
    with pytest.raises(sc.DimensionError):
        save_to_buffer(da, coord=coord_name)


@given(da=one_dim_data_arrays(), header=headers())
def test_can_set_header(da, header):
    buffer = save_to_buffer(da, coord=next(iter(da.coords)), header=header)
    commented_header = (
        '\n'.join(f'# {line}' for line in header.splitlines()) if header else ''
    )
    assert buffer.getvalue().startswith(commented_header)


@given(
    da=one_dim_data_arrays(), coord_name=st.text(string.ascii_letters + string.digits)
)
def test_generated_header_includes_coord_name_and_units(da, coord_name):
    # Detecting the coord name in the file can be tricky if it can contain arbitrary
    # characters (like '\n' or '#') because lines in the header start with '# '.
    initial_name = next(iter(da.coords))
    da.coords[coord_name] = da.coords.pop(initial_name)

    buffer = save_to_buffer(da, coord=coord_name)
    assert coord_name in buffer.getvalue()
    if da.coords[coord_name].unit is not None:
        assert str(da.coords[coord_name].unit) in buffer.getvalue()
    if da.unit is not None:
        assert str(da.unit) in buffer.getvalue()


def test_loads_correct_values():
    file_contents = '''1 2 3
1.003 32.1 5
0.1111 0 2.1e-3
'''
    buffer = StringIO(file_contents)
    loaded = scn.io.load_xye(buffer, dim='my-dim', unit='one', coord_unit='us')
    expected = sc.DataArray(
        sc.array(
            dims=['my-dim'],
            values=[2, 32.1, 0],
            variances=np.power([3, 5, 2.1e-3], 2),
            unit='one',
        ),
        coords={
            'my-dim': sc.array(dims=['my-dim'], values=[1, 1.003, 0.1111], unit='us')
        },
    )
    assert sc.identical(loaded, expected)
