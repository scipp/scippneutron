# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import math
from string import Template
from typing import Dict, Tuple

import scipp as sc

from .disk_chopper import DiskChopper

_CHOPPER_TEMPLATE = Template(
    '''<svg
    version="1.1"
    width="${image_size}"
    height="${image_size}"
    xmlns="http://www.w3.org/2000/svg">
${elements}
</svg>
'''
)
_DISK_TEMPLATE = Template(
    '<path d="${path}" fill="gray" ' 'stroke="#c00000" stroke-width="2"/>'
)
_EDGE_MARK_TEMPLATE = Template(
    '<path d="${path}" stroke="black" stroke-width="2" ' 'stroke-dasharray="5,5"/>'
)
_EDGE_LABEL_TEMPLATE = Template(
    '<text x="${x}" y="${y}" text-anchor="${anchor}">' '${label}</text>'
)
_TDC_MARK_TEMPLATE = Template(
    '''<polygon points="${points}"/>
<path d="${path}" stroke="blue" stroke-width="2" stroke-dasharray="5,5"/>'''
)


def _combine_slits(chopper: DiskChopper) -> sc.DataArray:
    slits = sc.DataArray(
        sc.arange('slit', chopper.slits, unit=None),
        coords={
            'edge': chopper.slit_edges,
            'height': chopper.slit_height,
        },
    )
    return sc.sort(slits, key=slits.coords['edge']['edge', 0])


def _slit_edges_for_drawing(
    chopper: DiskChopper, edge: sc.Variable
) -> Tuple[sc.Variable, sc.Variable, bool]:
    # begin and end are in the order encountered when tracing a circle from TDC
    # in counter-clockwise direction.
    if chopper.rotation_speed < sc.scalar(0, unit=chopper.rotation_speed.unit):
        begin, end = edge
        return begin, end, True
    else:
        end, begin = edge
        return begin, end, False


def _tdc_marker(image_size: int) -> str:
    bottom = image_size // 2, 11
    left = image_size // 2 - 5, 1
    right = image_size // 2 + 5, 1
    return _TDC_MARK_TEMPLATE.substitute(
        points=f'{bottom[0]},{bottom[1]} {left[0]},{left[1]} {right[0]},{right[1]}',
        path=f'M{image_size // 2} {image_size // 2} L{bottom[0]} {bottom[1]}',
    )


def draw_chopper(chopper: DiskChopper, image_size: int) -> str:
    scale = image_size / 2 / chopper.radius * 0.99
    angle = sc.scalar(0.0, unit='rad')
    outer = True
    radius_scale = 0.8
    radius = chopper.radius * radius_scale

    def to_svg_coord(x: sc.Variable) -> float:
        # SVG coords have origin in top left
        # object coords have origin in center
        return float(x * scale) + image_size // 2

    def to_svg_length(x: sc.Variable) -> float:
        return float(x * scale)

    def polar_to_svg_coords(
        *, radius: sc.Variable, angle: sc.Variable
    ) -> Tuple[float, float]:
        x = to_svg_coord(radius * -sc.sin(angle))
        y = image_size - to_svg_coord(radius * sc.cos(angle))
        return x, y

    def move_to(*, radius: sc.Variable, angle: sc.Variable) -> str:
        x, y = polar_to_svg_coords(radius=radius, angle=angle)
        return f'M{x:.3f} {y:.3f}'

    def trace_arc(a: sc.Variable) -> str:
        nonlocal angle
        a = a.to(unit='rad')
        x, y = polar_to_svg_coords(radius=radius, angle=a)
        r = to_svg_length(radius)
        large = 1 if (a - angle) > sc.scalar(math.pi, unit='rad') else 0
        angle = a
        return f'A{r:.3f} {r:.3f} 0 {large} 0 {x:.3f} {y:.3f}'

    def trace_edge(h: sc.Variable) -> str:
        nonlocal outer, radius
        if outer:
            radius = (chopper.radius - h) * radius_scale
            outer = False
        else:
            radius = chopper.radius * radius_scale
            outer = True
        x, y = polar_to_svg_coords(radius=radius, angle=angle)
        return f'L{x:.3f} {y:.3f}'

    def edge_mark(is_open: bool, idx: int) -> Tuple[str, Dict[str, str]]:
        x, y = polar_to_svg_coords(radius=chopper.radius, angle=angle)
        line = f'M{image_size // 2} {image_size // 2} L{x:.3f} {y:.3f}'
        anchor = 'end' if x > image_size // 2 else 'start'
        text = {
            'x': x,
            'y': y,
            'anchor': anchor,
            'label': ('open' if is_open else 'close') + str(idx),
        }
        return line, text

    slits = _combine_slits(chopper)

    p = [
        move_to(radius=chopper.radius * radius_scale, angle=sc.scalar(0.0, unit='rad'))
    ]
    edge_marks = []
    for slit in slits:
        begin, end, open_first = _slit_edges_for_drawing(chopper, slit.coords['edge'])
        p.append(trace_arc(begin))
        p.append(trace_edge(slit.coords['height']))
        edge_marks.append(edge_mark(is_open=open_first, idx=slit.value))
        p.append(trace_arc(end))
        p.append(trace_edge(slit.coords['height']))
        edge_marks.append(edge_mark(is_open=not open_first, idx=slit.value))

    p.append(trace_arc(sc.scalar(2 * math.pi, unit='rad')))
    elements = [
        _DISK_TEMPLATE.substitute(path=' '.join(p)),
        *(
            _EDGE_MARK_TEMPLATE.substitute(path=path)
            + '\n'
            + _EDGE_LABEL_TEMPLATE.substitute(**label)
            for path, label in edge_marks
        ),
        _tdc_marker(image_size),
    ]

    return _CHOPPER_TEMPLATE.substitute(
        image_size=image_size, elements='\n'.join(elements)
    )


def draw_disk_chopper(chopper: DiskChopper, *, image_size: int) -> str:
    return draw_chopper(chopper, image_size)
