# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from typing import Literal

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
        da = da.drop_coords(list(set(da.coords) - {"position", dim}))
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


def instrument_view(
    data: sc.DataArray | sc.DataGroup | dict,
    dim: str | None = None,
    size: float | sc.Variable | None = None,
    pixel_size: float | sc.Variable | None = None,
    autoscale: bool = False,
    operation: Literal[
        'sum', 'mean', 'max', 'min', 'nansum', 'nanmean', 'nanmax', 'nanmin'
    ] = 'sum',
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
    operation:
        The reduction operation to be applied to the sliced dimensions. This is ``sum``
        by default.
    **kwargs:
        Additional arguments are forwarded to the scatter3d figure
        (see https://scipp.github.io/plopp/generated/plopp.scatter3d.html).
    """
    from plopp.plotting.slicer import Slicer
    from plopp.widgets import ClippingManager, ToggleTool, VBar

    data = _to_data_array(data, dim)

    if dim is not None:
        slicer = Slicer(pp.Node(data), keep=set(data.dims) - {dim}, operation=operation)
        to_scatter = slicer.output[0]
    else:
        to_scatter = pp.Node(data)

    size = size or pixel_size

    kwargs.setdefault('cbar', True)
    fig = pp.scatter3dfigure(
        to_scatter,
        x="x",
        y="y",
        z="z",
        size=sc.scalar(1.0, unit="cm") if size is None else size,
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
        widgets.append(slicer.slider)

        def _maybe_update_value_cut(_):
            if any(cut.kind == "v" for cut in clip_planes.cuts):
                clip_planes.update_state()

        slicer.slider.observe(_maybe_update_value_cut, names='value')

    fig.bottom_bar.add(VBar(widgets))

    return fig
