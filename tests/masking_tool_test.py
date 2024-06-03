# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc
from matplotlib.colors import to_hex

from scippneutron import MaskingTool


def make_data(npoints=50000, scale=10.0, seed=1):
    rng = np.random.default_rng(seed)
    position = scale * rng.standard_normal(size=[npoints, 2])
    values = np.linalg.norm(position, axis=1)
    return sc.DataArray(
        data=sc.array(dims=['row'], values=values, unit='K'),
        coords={
            'x': sc.array(dims=['row'], unit='m', values=position[:, 0]),
            'y': sc.array(dims=['row'], unit='m', values=position[:, 1]),
        },
    ).hist(y=300, x=300)


def test_mask_2d_with_rectangles(_use_ipympl):
    da = make_data()
    masking_tool = MaskingTool(da)
    r, _, _ = masking_tool.controls

    r.value = True
    r._tool.click(-10, -10)
    r._tool.click(5, 6)

    masks = masking_tool.get_masks()
    assert len(masks) == 1
    mask = next(iter(masks.values()))
    assert len(mask) == 2  # 2 dimensions
    assert sc.identical(mask['x'], sc.array(dims=['x'], values=[-10.0, 5.0], unit='m'))
    assert sc.identical(mask['y'], sc.array(dims=['y'], values=[-10.0, 6.0], unit='m'))

    r._tool.click(10, -20)
    r._tool.click(15, 30)

    masks = masking_tool.get_masks()
    assert len(masks) == 2
    mask = list(masks.values())[1]
    assert len(mask) == 2  # 2 dimensions
    assert sc.identical(mask['x'], sc.array(dims=['x'], values=[10.0, 15.0], unit='m'))
    assert sc.identical(mask['y'], sc.array(dims=['y'], values=[-20.0, 30.0], unit='m'))


def test_mask_2d_with_vspans(_use_ipympl):
    da = make_data()
    masking_tool = MaskingTool(da)
    _, v, _ = masking_tool.controls

    v.value = True
    v._tool.click(-20, 0)
    v._tool.click(1, 0)

    masks = masking_tool.get_masks()
    assert len(masks) == 1
    mask = next(iter(masks.values()))
    assert len(mask) == 1  # 1 dimension
    assert sc.identical(mask['x'], sc.array(dims=['x'], values=[-20.0, 1.0], unit='m'))

    v._tool.click(21, 0)
    v._tool.click(27, 0)

    masks = masking_tool.get_masks()
    assert len(masks) == 2
    mask = list(masks.values())[1]
    assert len(mask) == 1  # 1 dimension
    assert sc.identical(mask['x'], sc.array(dims=['x'], values=[21.0, 27.0], unit='m'))


def test_mask_2d_with_hspans(_use_ipympl):
    da = make_data()
    masking_tool = MaskingTool(da)
    _, _, h = masking_tool.controls

    h.value = True
    h._tool.click(0, -30)
    h._tool.click(0, 1)

    masks = masking_tool.get_masks()
    assert len(masks) == 1
    mask = next(iter(masks.values()))
    assert len(mask) == 1  # 1 dimension
    assert sc.identical(mask['y'], sc.array(dims=['y'], values=[-30.0, 1.0], unit='m'))

    h._tool.click(0, -10)
    h._tool.click(0, 17)

    masks = masking_tool.get_masks()
    assert len(masks) == 2
    mask = list(masks.values())[1]
    assert len(mask) == 1  # 1 dimension
    assert sc.identical(mask['y'], sc.array(dims=['y'], values=[-10.0, 17.0], unit='m'))


def test_mask_2d_rectangles_and_spans(_use_ipympl):
    da = make_data()
    masking_tool = MaskingTool(da)
    r, v, h = masking_tool.controls

    r.value = True
    r._tool.click(-10, -10)
    r._tool.click(5, 6)

    v.value = True
    v._tool.click(-20, 0)
    v._tool.click(1, 0)

    h.value = True
    h._tool.click(0, -30)
    h._tool.click(0, 1)

    masks = list(masking_tool.get_masks().values())
    assert len(masks) == 3

    mask = masks[0]
    assert len(mask) == 2  # 2 dimensions
    assert sc.identical(mask['x'], sc.array(dims=['x'], values=[-10.0, 5.0], unit='m'))
    assert sc.identical(mask['y'], sc.array(dims=['y'], values=[-10.0, 6.0], unit='m'))

    mask = masks[1]
    assert len(mask) == 1  # 1 dimension
    assert sc.identical(mask['x'], sc.array(dims=['x'], values=[-20.0, 1.0], unit='m'))

    mask = masks[2]
    assert len(mask) == 1  # 1 dimension
    assert sc.identical(mask['y'], sc.array(dims=['y'], values=[-30.0, 1.0], unit='m'))


def test_mask_1d(_use_ipympl):
    da = make_data().sum('y')
    masking_tool = MaskingTool(da)
    _, v, _ = masking_tool.controls

    v.value = True
    v._tool.click(-20, 0)
    v._tool.click(1, 0)

    masks = masking_tool.get_masks()
    assert len(masks) == 1
    mask = next(iter(masks.values()))
    assert len(mask) == 1  # 1 dimension
    assert sc.identical(mask['x'], sc.array(dims=['x'], values=[-20.0, 1.0], unit='m'))

    v._tool.click(21, 0)
    v._tool.click(27, 0)

    masks = masking_tool.get_masks()
    assert len(masks) == 2
    mask = list(masks.values())[1]
    assert len(mask) == 1  # 1 dimension
    assert sc.identical(mask['x'], sc.array(dims=['x'], values=[21.0, 27.0], unit='m'))


def test_mask_color(_use_ipympl):
    da = make_data()
    color = to_hex('red')
    masking_tool = MaskingTool(da, color=color)
    r, v, _ = masking_tool.controls

    r.value = True
    r._tool.click(-10, -10)
    r._tool.click(5, 6)

    shape = r._tool.children[0]
    assert to_hex(shape.facecolor) == color
    assert to_hex(shape.edgecolor) == color

    v.value = True
    v._tool.click(-20, 0)
    v._tool.click(1, 0)

    shape = v._tool.children[0]
    assert to_hex(shape.facecolor) == color
    assert to_hex(shape.edgecolor) == color
