# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# flake8: noqa: E501

"""SVG rendering for choppers."""

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
    '<path d="${path}" fill="#e0e0e0" ' 'stroke="#a0a0a0" stroke-width="2"/>'
)
_EDGE_MARK_TEMPLATE = Template(
    '<path d="${path}" stroke="#707070" stroke-width="2" stroke-dasharray="5,5"/>'
)
_EDGE_LABEL_TEMPLATE = Template(
    '<text x="${x}" y="${y}" text-anchor="${anchor}" font-family="sans-serif" '
    'dominant-baseline="ideographic" fill="#707070">'
    '${label}</text>'
)
_TDC_MARK_TEMPLATE = Template(
    '''<polygon fill="#0065ac" points="${points}"/>
<path d="${path}" stroke="#0065ac" stroke-width="2" stroke-dasharray="10,3"/>
<text x="${text_x}" y="${text_y}" text-anchor="${anchor}" dominant-baseline="middle" font-family="sans-serif" fill="#0065ac">TDC</text>'''
)
_BEAM_POSITION_TEMPLATE = Template(
    '''<path d="${path}" stroke="#cc5e14" stroke-width="2"/>
<text x="${text_x}" y="${text_y}" text-anchor="middle" dominant-baseline="middle" font-family="sans-serif" fill="#cc5e14">Beam Pos</text>'''
)


def _rotation_arrow(*, image_size: int, clockwise: bool) -> str:
    # The arrow was drawn with Inkscape at an image size of 400.
    s = image_size / 400
    flip, shift = (-1, image_size) if clockwise else (1, 0)
    return f'''<g id="rotation-arrow" transform="matrix({flip*s},0,0,{s},{shift},0)">
<path fill="#606060"
 d="m 190.46289,4.140625 c -19.2285,0.9736749 -38.20846,4.7790001 -56.32617,11.292969 l 0.89453,2.490234 C 152.90409,11.497902 171.62709,7.7437179 190.5957,6.7832031 Z"/>
<path fill="#606060" stroke-linecap="round"
 d="m 138.29883,5.5019531 a 1.322835,1.322835 0 0 0 -0.75,0.6816406 l -5.34766,11.3515623 11.35156,5.347656 A 1.322835,1.322835 0 0 0 145.3125,22.25 1.322835,1.322835 0 0 0 144.67969,20.488281 l -8.95703,-4.21875 4.21875,-8.9589841 a 1.322835,1.322835 0 0 0 -0.63282,-1.7597657 1.322835,1.322835 0 0 0 -1.00976,-0.048828 z"/>
</g>'''


def _combine_slits(chopper: DiskChopper) -> sc.DataArray:
    slits = sc.DataArray(
        sc.arange('slit', chopper.slits, unit=None),
        coords={
            'edge': chopper.slit_edges,
            'height': chopper.slit_height,
        },
    )
    return sc.sort(slits, key=slits.coords['edge']['edge', 0])


def _is_clockwise(chopper: DiskChopper) -> bool:
    return bool(chopper.rotation_speed < sc.scalar(0, unit=chopper.rotation_speed.unit))


def _slit_edges_for_drawing(
    chopper: DiskChopper, edge: sc.Variable
) -> Tuple[sc.Variable, sc.Variable, bool]:
    # begin and end are in the order encountered when tracing a circle from TDC
    # in counter-clockwise direction.
    if _is_clockwise(chopper):
        begin, end = edge
        return begin, end, True
    else:
        end, begin = edge
        return begin, end, False


def _tdc_marker(*, image_size: int, chopper: DiskChopper) -> str:
    bottom = image_size // 2, 11
    left = image_size // 2 - 5, 1
    right = image_size // 2 + 5, 1
    return _TDC_MARK_TEMPLATE.substitute(
        points=f'{bottom[0]},{bottom[1]} {left[0]},{left[1]} {right[0]},{right[1]}',
        path=f'M{image_size // 2} {image_size // 2} L{bottom[0]} {bottom[1]}',
        text_x=image_size // 2 + 7 * (-1 if _is_clockwise(chopper) else 1),
        text_y=11,
        anchor='end' if _is_clockwise(chopper) else 'start',
    )


def draw_disk_chopper(chopper: DiskChopper, *, image_size: int) -> str:
    """Generate an SVG image for a chopper."""

    # This functions and its sub-functions mimic an SVG path in that they maintain
    # a state that gets updated with every segment that gets added to the path.
    # This way, they trace out the chopper disk and its slits.
    #
    # There are two coordinate systems.
    # - Object coords: The chopper's own coordinate system in which its radius and
    #                  slit height are defined.
    #                  The origin is in the center of the chopper.
    # - SVG coords: The coords used for drawing in a box of size
    #               ``image_size x image_size``.
    #               The origin is in the top-left.
    #
    # ``scale`` maps from object to svg lengths.
    # In addition, ``radius_scale`` is applied to the disk radius to make room for
    # additional elements around the disk.
    # ``to_svg_coord`` maps a single x or y object coord to an SVG coord.
    # ``polar_to_svg_coords`` maps polar coordinates to cartesian SVG coordinates.
    # It applies a pi/2 counterclockwise rotation in order to position angle=0
    # at the top as chopper angles are defined in terms of the top dead center sensor.

    scale = image_size / 2 / chopper.radius * 0.99
    radius_scale = 0.8

    def to_svg_coord(x: sc.Variable) -> float:
        return float(x * scale) + image_size // 2

    def to_svg_length(x: sc.Variable) -> float:
        return float(x * scale)

    def polar_to_svg_coords(*, r: sc.Variable, a: sc.Variable) -> Tuple[float, float]:
        x = to_svg_coord(r * -sc.sin(a))
        y = image_size - to_svg_coord(r * sc.cos(a))
        return x, y

    # The state of the disk tracer.
    angle = sc.scalar(0.0, unit='rad')
    outer = True  # Currently tracing the outer circle or slit?
    radius = chopper.radius * radius_scale

    def move_to(*, r: sc.Variable, a: sc.Variable) -> str:
        nonlocal radius, angle
        a = a.to(unit='rad')
        radius = r
        angle = a
        x, y = polar_to_svg_coords(r=r, a=a)
        return f'M{x:.3f} {y:.3f}'

    def trace_arc(a: sc.Variable) -> str:
        nonlocal angle
        a = a.to(unit='rad')
        x, y = polar_to_svg_coords(r=radius, a=a)
        r = to_svg_length(radius)
        pi = sc.scalar(math.pi, unit='rad')
        large = 1 if (a - angle) > pi else 0
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
        x, y = polar_to_svg_coords(r=radius, a=angle)
        return f'L{x:.3f} {y:.3f}'

    def edge_mark(is_open: bool, idx: int) -> Tuple[str, Dict[str, str]]:
        x, y = polar_to_svg_coords(r=chopper.radius, a=angle)
        line = f'M{image_size // 2} {image_size // 2} L{x:.3f} {y:.3f}'
        anchor = 'end' if x > 3 * image_size // 4 else 'start'
        text = {
            'x': x,
            'y': y,
            'anchor': anchor,
            'label': ('open' if is_open else 'close') + str(idx),
        }
        return line, text

    def beam_pos(beam_position: sc.Variable) -> str:
        x, y = polar_to_svg_coords(r=chopper.radius * radius_scale, a=beam_position)
        return _BEAM_POSITION_TEMPLATE.substitute(
            path=f'M{image_size // 2} {image_size // 2} L{x} {y}',
            text_x=x,
            text_y=y,
        )

    start_angle = None
    disk_path = []
    edge_marks = []
    for slit in _combine_slits(chopper):
        begin, end, open_first = _slit_edges_for_drawing(chopper, slit.coords['edge'])

        if start_angle is None:
            start_angle = begin.to(unit='rad')
            disk_path.append(move_to(r=chopper.radius * radius_scale, a=start_angle))
        else:
            disk_path.append(trace_arc(begin))
        disk_path.append(trace_edge(slit.coords['height']))
        edge_marks.append(edge_mark(is_open=open_first, idx=slit.value))
        disk_path.append(trace_arc(end))
        disk_path.append(trace_edge(slit.coords['height']))
        edge_marks.append(edge_mark(is_open=not open_first, idx=slit.value))

    if start_angle < angle:
        disk_path.append(trace_arc(sc.scalar(2 * math.pi, unit='rad') + start_angle))
    else:
        disk_path.append(trace_arc(start_angle))

    elements = [
        _DISK_TEMPLATE.substitute(path=' '.join(disk_path)),
        *(
            _EDGE_MARK_TEMPLATE.substitute(path=path)
            + '\n'
            + _EDGE_LABEL_TEMPLATE.substitute(**label)
            for path, label in edge_marks
        ),
        _tdc_marker(image_size=image_size, chopper=chopper),
        _rotation_arrow(image_size=image_size, clockwise=_is_clockwise(chopper)),
    ]
    if (beam_position := chopper.beam_position) is not None:
        elements.append(beam_pos(beam_position))
    elements.append(f'<circle cx="{image_size/2}" cy="{image_size/2}" r="5"/>')
    return _CHOPPER_TEMPLATE.substitute(
        image_size=image_size, elements='\n'.join(elements)
    )
