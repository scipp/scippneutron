# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import dataclasses
import uuid
from string import Template
from typing import TYPE_CHECKING

import scipp as sc

from ._resources import disk_chopper_repr_template, disk_chopper_style

try:
    from scipp.visualization.formatting_datagroup_html import (
        _datagroup_detail,
        _format_shape,
        load_dg_repr_tpl,
    )
except ImportError:
    from scipp.html.formatting_datagroup_html import (
        _datagroup_detail,
        _format_shape,
        load_dg_repr_tpl,
    )

if TYPE_CHECKING:
    from ..chopper import DiskChopper
else:
    DiskChopper = object


def disk_chopper_html_repr(chopper: DiskChopper) -> str:
    field_repr = _datagroup_repr(
        sc.DataGroup(
            {
                field.name: getattr(chopper, field.name)
                for field in dataclasses.fields(chopper)
                if not field.name.startswith('_')
            }
        ),
        type_name="DiskChopper",
        max_length_before_fold=16,
    )
    try:
        image = chopper.make_svg()
    except RuntimeError:
        image = ""
    return disk_chopper_repr_template().substitute(
        style_sheet=disk_chopper_style(), fields=field_repr, image=image
    )


def _datagroup_repr(
    dg: sc.DataGroup, *, type_name: str, max_length_before_fold: int = 15
) -> str:
    checkbox_status = "checked" if len(dg) < max_length_before_fold else ''
    header_id = "group-view-" + str(uuid.uuid4())
    details = _datagroup_detail(dg)
    html = Template(load_dg_repr_tpl())
    return html.substitute(
        style_sheet='',
        header_id=header_id,
        checkbox_status=checkbox_status,
        obj_type=type_name,
        shape_repr=_format_shape(dg, br_at=200),
        details=details,
    )
