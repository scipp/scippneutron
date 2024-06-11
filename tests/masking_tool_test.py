# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import json

import numpy as np
import pytest
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


@pytest.mark.usefixtures('_use_ipympl')
def test_mask_2d_with_rectangles():
    da = make_data()
    masking_tool = MaskingTool(da)
    r, _, _ = masking_tool.controls

    r.value = True
    r._tool.click(-10.0, -10.0)
    r._tool.click(5.0, 6.0)

    masks = masking_tool.get_masks()
    assert len(masks) == 1
    mask = next(iter(masks.values()))
    assert mask["kind"] == "rectangle"
    bounds = mask["bounds"]
    assert len(bounds) == 2  # 2 dimensions
    assert bounds['x']["min"] == {"value": -10.0, "unit": "m"}
    assert bounds['x']["max"] == {"value": 5.0, "unit": "m"}
    assert bounds['y']["min"] == {"value": -10.0, "unit": "m"}
    assert bounds['y']["max"] == {"value": 6.0, "unit": "m"}

    r._tool.click(10.0, -20.0)
    r._tool.click(15.0, 30.0)

    masks = masking_tool.get_masks()
    assert len(masks) == 2
    mask = list(masks.values())[1]
    assert mask["kind"] == "rectangle"
    bounds = mask["bounds"]
    assert len(bounds) == 2  # 2 dimensions
    assert bounds['x']["min"] == {"value": 10.0, "unit": "m"}
    assert bounds['x']["max"] == {"value": 15.0, "unit": "m"}
    assert bounds['y']["min"] == {"value": -20.0, "unit": "m"}
    assert bounds['y']["max"] == {"value": 30.0, "unit": "m"}


@pytest.mark.usefixtures('_use_ipympl')
def test_mask_2d_with_vspans():
    da = make_data()
    masking_tool = MaskingTool(da)
    _, v, _ = masking_tool.controls

    v.value = True
    v._tool.click(-20.0, 0.0)
    v._tool.click(1.0, 0.0)

    masks = masking_tool.get_masks()
    assert len(masks) == 1
    mask = next(iter(masks.values()))
    assert mask["kind"] == "vspan"
    bounds = mask["bounds"]
    assert len(bounds) == 1  # 1 dimension
    assert bounds['x']["min"] == {"value": -20.0, "unit": "m"}
    assert bounds['x']["max"] == {"value": 1.0, "unit": "m"}

    v._tool.click(21.0, 0.0)
    v._tool.click(27.0, 0.0)

    masks = masking_tool.get_masks()
    assert len(masks) == 2
    mask = list(masks.values())[1]
    assert mask["kind"] == "vspan"
    bounds = mask["bounds"]
    assert len(bounds) == 1  # 1 dimension
    assert bounds['x']["min"] == {"value": 21.0, "unit": "m"}
    assert bounds['x']["max"] == {"value": 27.0, "unit": "m"}


@pytest.mark.usefixtures('_use_ipympl')
def test_mask_2d_with_hspans():
    da = make_data()
    masking_tool = MaskingTool(da)
    _, _, h = masking_tool.controls

    h.value = True
    h._tool.click(0.0, -30.0)
    h._tool.click(0.0, 1.0)

    masks = masking_tool.get_masks()
    assert len(masks) == 1
    mask = next(iter(masks.values()))
    assert mask["kind"] == "hspan"
    bounds = mask["bounds"]
    assert len(bounds) == 1  # 1 dimension
    assert bounds['y']["min"] == {"value": -30.0, "unit": "m"}
    assert bounds['y']["max"] == {"value": 1.0, "unit": "m"}

    h._tool.click(0.0, -10.0)
    h._tool.click(0.0, 17.0)

    masks = masking_tool.get_masks()
    assert len(masks) == 2
    mask = list(masks.values())[1]
    assert mask["kind"] == "hspan"
    bounds = mask["bounds"]
    assert len(bounds) == 1  # 1 dimension
    assert bounds['y']["min"] == {"value": -10.0, "unit": "m"}
    assert bounds['y']["max"] == {"value": 17.0, "unit": "m"}


@pytest.mark.usefixtures('_use_ipympl')
def test_mask_2d_rectangles_and_spans():
    da = make_data()
    masking_tool = MaskingTool(da)
    r, v, h = masking_tool.controls

    r.value = True
    r._tool.click(-10.0, -10.0)
    r._tool.click(5.0, 6.0)

    v.value = True
    v._tool.click(-20.0, 0.0)
    v._tool.click(1.0, 0.0)

    h.value = True
    h._tool.click(0.0, -30.0)
    h._tool.click(0.0, 1.0)

    masks = list(masking_tool.get_masks().values())
    assert len(masks) == 3

    mask = masks[0]
    assert mask["kind"] == "rectangle"
    bounds = mask["bounds"]
    assert len(bounds) == 2  # 2 dimensions
    assert bounds['x']["min"] == {"value": -10.0, "unit": "m"}
    assert bounds['x']["max"] == {"value": 5.0, "unit": "m"}
    assert bounds['y']["min"] == {"value": -10.0, "unit": "m"}
    assert bounds['y']["max"] == {"value": 6.0, "unit": "m"}

    mask = masks[1]
    assert mask["kind"] == "vspan"
    bounds = mask["bounds"]
    assert len(bounds) == 1  # 1 dimension
    assert bounds['x']["min"] == {"value": -20.0, "unit": "m"}
    assert bounds['x']["max"] == {"value": 1.0, "unit": "m"}

    mask = masks[2]
    assert mask["kind"] == "hspan"
    bounds = mask["bounds"]
    assert len(bounds) == 1  # 1 dimension
    assert bounds['y']["min"] == {"value": -30.0, "unit": "m"}
    assert bounds['y']["max"] == {"value": 1.0, "unit": "m"}


@pytest.mark.usefixtures('_use_ipympl')
def test_mask_1d():
    da = make_data().sum('y')
    masking_tool = MaskingTool(da)
    _, v, _ = masking_tool.controls

    v.value = True
    v._tool.click(-20.0, 0.0)
    v._tool.click(1.0, 0.0)

    masks = masking_tool.get_masks()
    assert len(masks) == 1
    mask = next(iter(masks.values()))
    assert mask["kind"] == "vspan"
    bounds = mask["bounds"]
    assert len(bounds) == 1  # 1 dimension
    assert bounds['x']["min"] == {"value": -20.0, "unit": "m"}
    assert bounds['x']["max"] == {"value": 1.0, "unit": "m"}

    v._tool.click(21.0, 0.0)
    v._tool.click(27.0, 0.0)

    masks = masking_tool.get_masks()
    assert len(masks) == 2
    mask = list(masks.values())[1]
    assert mask["kind"] == "vspan"
    bounds = mask["bounds"]
    assert len(bounds) == 1  # 1 dimension
    assert bounds['x']["min"] == {"value": 21.0, "unit": "m"}
    assert bounds['x']["max"] == {"value": 27.0, "unit": "m"}


@pytest.mark.usefixtures('_use_ipympl')
def test_mask_color():
    da = make_data()
    color = to_hex('red')
    masking_tool = MaskingTool(da, color=color)
    r, v, _ = masking_tool.controls

    r.value = True
    r._tool.click(-10.0, -10.0)
    r._tool.click(5.0, 6.0)

    shape = r._tool.children[0]
    assert to_hex(shape.facecolor) == color
    assert to_hex(shape.edgecolor) == color

    v.value = True
    v._tool.click(-20.0, 0.0)
    v._tool.click(1.0, 0.0)

    shape = v._tool.children[0]
    assert to_hex(shape.facecolor) == color
    assert to_hex(shape.edgecolor) == color


@pytest.mark.usefixtures('_use_ipympl')
def test_save_masks(tmp_path):
    da = make_data()
    masking_tool = MaskingTool(da)
    r, v, h = masking_tool.controls

    r.value = True
    r._tool.click(-10.0, -10.0)
    r._tool.click(5.0, 6.0)

    v.value = True
    v._tool.click(-20.0, 0.0)
    v._tool.click(1.0, 0.0)

    h.value = True
    h._tool.click(0.0, -30.0)
    h._tool.click(0.0, 1.0)

    file = tmp_path / ".json"
    masking_tool.save_masks(file)

    with open(file) as f:
        loaded_masks = json.load(f)

    assert len(loaded_masks) == 3

    masks = list(loaded_masks.values())

    mask = masks[0]
    assert mask["kind"] == "rectangle"
    bounds = mask["bounds"]
    assert len(bounds) == 2  # 2 dimensions
    assert bounds['x']["min"] == {"value": -10.0, "unit": "m"}
    assert bounds['x']["max"] == {"value": 5.0, "unit": "m"}
    assert bounds['y']["min"] == {"value": -10.0, "unit": "m"}
    assert bounds['y']["max"] == {"value": 6.0, "unit": "m"}

    mask = masks[1]
    assert mask["kind"] == "vspan"
    bounds = mask["bounds"]
    assert len(bounds) == 1  # 1 dimension
    assert bounds['x']["min"] == {"value": -20.0, "unit": "m"}
    assert bounds['x']["max"] == {"value": 1.0, "unit": "m"}

    mask = masks[2]
    assert mask["kind"] == "hspan"
    bounds = mask["bounds"]
    assert len(bounds) == 1  # 1 dimension
    assert bounds['y']["min"] == {"value": -30.0, "unit": "m"}
    assert bounds['y']["max"] == {"value": 1.0, "unit": "m"}


@pytest.mark.usefixtures('_use_ipympl')
def test_masking_tool_raises_with_3d_data():
    da = make_data() * sc.arange('z', 3)
    with pytest.raises(
        ValueError, match="The masking tool can only handle 1D or 2D data."
    ):
        MaskingTool(da)
