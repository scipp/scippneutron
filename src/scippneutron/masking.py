# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import json
import os
from functools import partial
from typing import TYPE_CHECKING

import plopp as pp
import scipp as sc
from matplotlib.colors import to_rgb
from plopp.core.typing import FigureLike

if TYPE_CHECKING:
    from mpltoolbox import Patch


def _define_rect_mask(da: sc.DataArray, rect_info: dict) -> sc.Variable:
    """
    Function that creates a mask inside the area
    covered by the rectangle.
    """
    x = rect_info["x"]
    y = rect_info["y"]
    xcoord = da.coords[x["dim"]]
    ycoord = da.coords[y["dim"]]
    return (
        (xcoord >= x["min"])
        & (xcoord <= x["max"])
        & (ycoord >= y["min"])
        & (ycoord <= y["max"])
    )


def _get_rect_info(artist: Patch, figure: FigureLike) -> dict:
    """
    Convert the raw rectangle info to a dict containing the dimensions of
    each axis, and values with units.
    """
    x1 = artist.xy[0]
    x2 = artist.xy[0] + artist.width
    y1 = artist.xy[1]
    y2 = artist.xy[1] + artist.height
    return lambda: {
        "x": {
            "dim": figure.canvas.dims["x"],
            "min": sc.scalar(min(x1, x2), unit=figure.canvas.units["x"]),
            "max": sc.scalar(max(x1, x2), unit=figure.canvas.units["x"]),
        },
        "y": {
            "dim": figure.canvas.dims["y"],
            "min": sc.scalar(min(y1, y2), unit=figure.canvas.units["y"]),
            "max": sc.scalar(max(y1, y2), unit=figure.canvas.units["y"]),
        },
    }


def _define_span_mask(da: sc.DataArray, span_info: dict) -> sc.Variable:
    info = next(iter(span_info.values()))
    coord = da.coords[info["dim"]]
    return (coord >= info["min"]) & (coord <= info["max"])


def _get_vspan_info(artist: Patch, figure: FigureLike) -> dict:
    x1 = artist.left
    x2 = artist.right
    return lambda: {
        "x": {
            "dim": figure.canvas.dims["x"],
            "min": sc.scalar(min(x1, x2), unit=figure.canvas.units["x"]),
            "max": sc.scalar(max(x1, x2), unit=figure.canvas.units["x"]),
        }
    }


def _get_hspan_info(artist: Patch, figure: FigureLike) -> dict:
    y1 = artist.bottom
    y2 = artist.top
    return lambda: {
        "y": {
            "dim": figure.canvas.dims["y"],
            "min": sc.scalar(min(y1, y2), unit=figure.canvas.units["y"]),
            "max": sc.scalar(max(y1, y2), unit=figure.canvas.units["y"]),
        }
    }


def _apply_masks(da: sc.DataArray, *masks: sc.Variable) -> sc.DataArray:
    out = da.copy(deep=False)
    for i, mask in enumerate(masks):
        out.masks[str(i)] = mask
    return out


def _scalar_to_dict(scalar: sc.Variable) -> dict:
    return {"value": scalar.value, "unit": str(scalar.unit)}


class MaskingTool:
    def __init__(self, data: sc.DataArray, color='magenta', **kwargs):
        """
        Interactive masking tool for 1D and 2D data.
        The tool will display a figure with the data and allow the user to
        draw rectangles, horizontal spans, and vertical spans to create masks, using
        buttons in the top bar of the figure.

        Parameters
        ----------
        data:
            The data to be masked.
        color:
            The color of the shapes drawn by the user.
        kwargs:
            Additional keyword arguments passed to the figure constructor.
        """
        import ipywidgets as ipw
        from mpltoolbox import Hspans, Rectangles, Vspans
        from plopp.widgets import DrawingTool, style

        # Convert potential bin edge coords to midpoints
        da = data.copy(deep=False)
        for dim, coord in data.coords.items():
            if data.coords.is_edges(dim):
                da.coords[dim] = sc.midpoints(coord)

        ndim = da.ndim
        self.data_node = pp.Node(da)
        self.masking_node = pp.Node(_apply_masks, self.data_node)
        self.fig = {1: pp.linefigure, 2: pp.imagefigure}[ndim](
            self.masking_node, **kwargs
        )

        common = {
            "figure": self.fig,
            "input_node": self.data_node,
            "destination": self.masking_node,
        }

        col_args = {"edgecolor": color, "facecolor": (*to_rgb(color), 0.05)}

        rects = DrawingTool(
            tool=partial(Rectangles, **col_args),
            get_artist_info=_get_rect_info,
            icon="vector-square",
            func=_define_rect_mask,
            disabled=ndim == 1,
            **common,
        )
        vspans = DrawingTool(
            tool=partial(Vspans, **col_args),
            get_artist_info=_get_vspan_info,
            icon="grip-lines-vertical",
            func=_define_span_mask,
            **common,
        )
        hspans = DrawingTool(
            tool=partial(Hspans, **col_args),
            get_artist_info=_get_hspan_info,
            icon="grip-lines",
            func=_define_span_mask,
            disabled=ndim == 1,
            **common,
        )
        self.controls = [rects, vspans, hspans]

        self.fig.top_bar.add(
            ipw.Label(
                "Add masks:",
                layout={
                    "display": "flex",
                    "justify_content": "flex-end",
                    "width": "110px",
                },
            )
        )
        for c in self.controls:
            self.fig.top_bar.add(c)
            c._tool._on_change.clear()
            c._tool.on_vertex_release(c.update_node)
            c._tool.on_drag_release(c.update_node)
            c._tool.on_remove(self.masking_node.notify_children)
            c.observe(self.toggle_button_states, names="value")

        self.filename = ipw.Text(placeholder="Save to file", layout={"width": "200px"})
        self.filename.observe(self.validate_filename, names="value")
        self.save_button = ipw.Button(
            icon="save", tooltip="Save to JSON", disabled=True, **style.BUTTON_LAYOUT
        )
        self.save_button.on_click(self.save_masks)
        self.toggle_visibility = ipw.ToggleButton(
            value=True, icon="eye-slash", tooltip="Hide shapes", **style.BUTTON_LAYOUT
        )
        self.toggle_visibility.observe(self.toggle_shape_visibility, names="value")
        self.fig.top_bar.add(ipw.HBox([], layout={"width": "40px"}))
        self.fig.top_bar.add(self.toggle_visibility)
        self.fig.top_bar.add(self.filename)
        self.fig.top_bar.add(self.save_button)

    def toggle_button_states(self, change: dict) -> None:
        if change["new"]:
            for c in self.controls:
                if c.value and c is not change["owner"]:
                    c.value = False

    def validate_filename(self, change: dict) -> None:
        path = change["new"]
        if os.path.exists(path):
            self.filename.style.background = "#ff9999"
            self.save_button.disabled = True
        else:
            self.filename.style.background = "#99ff99"
            self.save_button.disabled = False

    def toggle_shape_visibility(self, change: dict):
        for c in self.controls:
            for child in c._tool.children:
                child.set(visible=change["new"])
        self.toggle_visibility.icon = "eye-slash" if change["new"] else "eye"
        self.toggle_visibility.tooltip = (
            "Hide shapes" if change["new"] else "Show shapes"
        )
        self.fig.canvas.draw()

    def get_masks(self) -> dict:
        masks = {}
        mask_counter = 0
        for c in self.controls:
            for node in c._draw_nodes.values():
                info = node()
                mask_dims = "".join([axis["dim"] for axis in info.values()])
                mask_name = f"{mask_dims}_{mask_counter}"
                masks[mask_name] = {
                    axis["dim"]: {
                        "min": _scalar_to_dict(axis["min"]),
                        "max": _scalar_to_dict(axis["max"]),
                    }
                    for axis in info.values()
                }
                mask_counter += 1
        return masks

    def save_masks(self, _=None) -> None:
        with open('masks.json', 'w') as f:
            json.dump(self.get_masks(), f, indent=2)

    def _repr_mimebundle_(self, **kwargs) -> dict:
        return self.fig._repr_mimebundle_()
