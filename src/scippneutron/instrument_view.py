# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import plopp as pp
import scipp as sc
from plopp.core.typing import FigureLike


def _to_data_array(
    data: sc.DataArray | sc.DataGroup | dict, dim: str | None
) -> sc.DataArray:
    if isinstance(data, sc.DataArray):
        data = sc.DataGroup({"": data})
    pieces = []
    for da in data.values():
        da = da.drop_coords(set(da.coords) - {"position", dim})
        dims = list(da.dims)
        if (dim is not None) and (dim in dims):
            # Ensure that the dims to be flattened are contiguous
            da = da.transpose([d for d in dims if d != dim] + [dim])
            dims.remove(dim)
        flat = da.flatten(dims=dims, to="pixel")
        filtered = flat[sc.isfinite(flat.coords["position"])]
        pieces.append(
            filtered.assign_coords(
                {k: getattr(filtered.coords["position"].fields, k) for k in "xyz"}
            ).drop_coords("position")
        )
    return sc.concat(pieces, dim="pixel").squeeze()


def _slice_dim(
    da: sc.DataArray, slice_params: dict[str, tuple[int, int]]
) -> sc.DataArray:
    (params,) = slice_params.items()
    return da[params[0], params[1][0] : params[1][1] + 1].sum(params[0])


def instrument_view(
    data: sc.DataArray | sc.DataGroup | dict,
    dim: str | None = None,
    pixel_size: float | sc.Variable | None = None,
    autoscale: bool = False,
    **kwargs,
) -> FigureLike:
    """
    Three-dimensional visualization of the DREAM instrument.
    The instrument view is capable of slicing the input data with a slider widget along
    a dimension (e.g. ``tof``) by using the ``dim`` argument.

    Use the clipping tool to create cuts in 3d space, as well as according to data
    values.

    Parameters
    ----------
    data:
        Data to visualize. The data can be a single detector module (``DataArray``),
        or a group of detector modules (``dict`` or ``DataGroup``).
        The data must contain a ``position`` coordinate.
    dim:
        Dimension to use for the slider. No slider will be shown if this is None.
    pixel_size:
        Size of the pixels.
    autoscale:
        If ``True``, the color scale will be automatically adjusted to the data as it
        gets updated. This can be somewhat expensive with many pixels, so it is set to
        ``False`` by default.
    **kwargs:
        Additional arguments are forwarded to the scatter3d figure
        (see https://scipp.github.io/plopp/generated/plopp.scatter3d.html).
    """
    from ipywidgets import ToggleButtons
    from plopp.widgets import (
        ClippingManager,
        HBar,
        RangeSliceWidget,
        SliceWidget,
        ToggleTool,
        VBar,
    )

    data = _to_data_array(data, dim)

    if dim is not None:
        int_slicer = SliceWidget(data, dims=[dim])
        int_slider = int_slicer.controls[dim].slider
        int_slider.value = int_slider.min
        int_slider.layout = {"width": "42em"}

        range_slicer = RangeSliceWidget(data, dims=[dim])
        range_slider = range_slicer.controls[dim].slider
        range_slider.value = 0, data.sizes[dim]
        range_slider.layout = {"width": "42em"}

        def move_range(change):
            range_slider.value = (change["new"], change["new"])

        int_slider.observe(move_range, names='value')
        slider_toggler = ToggleButtons(
            options=["o-o", "-o-"],
            tooltips=['Range slider', 'Single slice slider'],
            style={"button_width": "3.2em"},
        )

        slicing_container = HBar([slider_toggler, range_slicer])

        def toggle_slider_mode(change):
            if change["new"] == "o-o":
                slicing_container.children = [slider_toggler, range_slicer]
            else:
                int_slider.value = int(0.5 * sum(range_slider.value))
                slicing_container.children = [slider_toggler, int_slicer]

        slider_toggler.observe(toggle_slider_mode, names='value')

        slider_node = pp.widget_node(range_slicer)
        to_scatter = pp.Node(_slice_dim, da=data, slice_params=slider_node)

    else:
        to_scatter = pp.Node(data)

    kwargs.setdefault('cbar', True)
    fig = pp.scatter3dfigure(
        to_scatter,
        x="x",
        y="y",
        z="z",
        pixel_size=1.0 * sc.Unit("cm") if pixel_size is None else pixel_size,
        autoscale=autoscale,
        **kwargs,
    )

    clip_planes = ClippingManager(fig)
    fig.toolbar['cut3d'] = ToggleTool(
        callback=clip_planes.toggle_visibility,
        icon='layer-group',
        tooltip='Hide/show spatial cutting tool',
    )
    widgets = [clip_planes]
    if dim is not None:
        widgets.append(slicing_container)

        def _maybe_update_value_cut(_):
            if any(cut.kind == "v" for cut in clip_planes.cuts):
                clip_planes.update_state()

        range_slicer.observe(_maybe_update_value_cut, names='value')

    fig.bottom_bar.add(VBar(widgets))

    return fig
