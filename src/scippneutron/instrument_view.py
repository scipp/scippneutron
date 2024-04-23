# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet and Owen Arnold

import numpy as np
import scipp as sc
from scipy.spatial.transform import Rotation as Rot

from ._utils import get_meta

try:
    import pythreejs as p3
except ImportError as ex:
    p3 = None
    _pythreejs_import_error = ex


def _create_text_sprite(position, bounding_box, display_text):
    # Position offset in y
    text_position = tuple(position.value + np.array([0, 0.8 * bounding_box[1], 0]))
    text = p3.TextTexture(string=display_text, color='black', size=300)
    text_material = p3.SpriteMaterial(map=text, transparent=True)
    size = 1.0
    return p3.Sprite(
        material=text_material, position=text_position, scale=[size, size, size]
    )


def _create_mesh(geometry, color, wireframe, position):
    if wireframe:
        edges = p3.EdgesGeometry(geometry)
        mesh = p3.LineSegments(
            geometry=edges, material=p3.LineBasicMaterial(color=color)
        )

    else:
        material = p3.MeshBasicMaterial(color=color)
        mesh = p3.Mesh(geometry=geometry, material=material)
    mesh.position = tuple(position.value)
    return mesh


def _box(position, display_text, bounding_box, color, wireframe, **kwargs):
    geometry = p3.BoxGeometry(
        width=bounding_box[0],
        height=bounding_box[1],
        depth=bounding_box[2],
        widthSegments=2,
        heightSegments=2,
        depthSegments=2,
    )

    mesh = _create_mesh(
        geometry=geometry, color=color, wireframe=wireframe, position=position
    )
    text_mesh = _create_text_sprite(
        position=position, bounding_box=bounding_box, display_text=display_text
    )
    return mesh, text_mesh


def _find_beam(det_com, pos):
    # Assume beam is axis aligned and follows largest axis
    # delta between component and detector COM
    beam_dir = np.argmax(np.abs(det_com.value - pos.value))
    beam = np.zeros(3)
    beam[beam_dir] = 1
    return beam


def _alignment_matrix(to_align, target):
    rot_axis = np.cross(to_align, target)
    magnitude = np.linalg.norm(to_align) * np.linalg.norm(target)
    axis_angle = np.arcsin(rot_axis / magnitude)
    return Rot.from_rotvec(axis_angle).as_matrix()


def _disk_chopper(position, display_text, bounding_box, color, wireframe, **kwargs):
    geometry = p3.CylinderGeometry(
        radiusTop=bounding_box[0] / 2,
        radiusBottom=bounding_box[0] / 2,
        height=bounding_box[0] / 100,
        radialSegments=24,
        heightSegments=12,
        openEnded=False,
        thetaStart=np.pi / 8,
        thetaLength=2 * np.pi - (np.pi / 8),
    )

    mesh = _create_mesh(
        geometry=geometry, color=color, wireframe=wireframe, position=position
    )
    beam = _find_beam(det_com=kwargs['det_center'], pos=position)
    disk_axis = np.array([0, 1, 0])  # Disk created with this axis
    rotation = _alignment_matrix(to_align=disk_axis, target=beam)
    mesh.setRotationFromMatrix(rotation.flatten())
    text_mesh = _create_text_sprite(
        position=position, bounding_box=bounding_box, display_text=display_text
    )
    return mesh, text_mesh


def _cylinder(position, display_text, bounding_box, color, wireframe, **kwargs):
    geometry = p3.CylinderGeometry(
        radiusTop=bounding_box[0] / 2,
        radiusBottom=bounding_box[0] / 2,
        height=bounding_box[1],
        radialSegments=12,
        heightSegments=12,
        openEnded=False,
        thetaStart=0,
        thetaLength=2.0 * np.pi,
    )
    mesh = _create_mesh(
        geometry=geometry, color=color, wireframe=wireframe, position=position
    )
    # Position label above cylinder
    text_mesh = _create_text_sprite(
        position=position, bounding_box=bounding_box, display_text=display_text
    )
    return mesh, text_mesh


def _unpack_to_scene(scene, items):
    if hasattr(items, "__iter__"):
        for item in items:
            scene.add(item)
    else:
        scene.add(items)


def _add_to_scene(
    position, scene, shape, display_text, bounding_box, color, wireframe, **kwargs
):
    _unpack_to_scene(
        scene,
        shape(
            position,
            display_text=display_text,
            bounding_box=bounding_box,
            color=color,
            wireframe=wireframe,
            **kwargs,
        ),
    )


def _furthest_component(det_center, scipp_obj, additional):
    distances = [
        sc.norm(settings["center"] - det_center).value
        for settings in list(additional.values())
    ]
    max_displacement = sorted(distances)[-1]
    return max_displacement


def _instrument_view_shape_types():
    return {"box": _box, "cylinder": _cylinder, "disk": _disk_chopper}


def _as_vector(var):
    if var.dtype == sc.DType.vector3:
        return var
    else:
        return sc.spatial.as_vectors(x=var, y=var, z=var)


def _plot_components(scipp_obj, components, positions_var, scene):
    det_center = sc.mean(positions_var)
    # Some scaling to set width according to distance from detector center
    shapes = _instrument_view_shape_types()
    for name, settings in components.items():
        type = settings["type"]
        size = _as_vector(settings["size"])
        component_position = settings["center"]
        color = settings.get("color", "#808080")
        wireframe = settings.get("wireframe", False)
        if type not in shapes:
            supported_shapes = ", ".join(shapes.keys())
            raise ValueError(
                f"Unknown shape: {type} requested for {name}. "
                f"Allowed values are: {supported_shapes}"
            )
        component_position = sc.to_unit(component_position, positions_var.unit)
        size = sc.to_unit(size, positions_var.unit)
        _add_to_scene(
            position=component_position,
            scene=scene,
            shape=shapes[type],
            display_text=name,
            bounding_box=tuple(size.values),
            color=color,
            wireframe=wireframe,
            det_center=det_center,
        )
    # Reset camera
    camera = _get_camera(scene)
    if camera:
        furthest_distance = _furthest_component(det_center, scipp_obj, components)
        camera.far = max(camera.far, furthest_distance * 5.0)


def _get_camera(scene):
    for child in scene.children:
        if isinstance(child, p3.PerspectiveCamera):
            return child
    return None


def instrument_view(
    scipp_obj, positions="position", pixel_size=None, components=None, **kwargs
):
    """Plot a 3D view of the instrument, using the `position` coordinate as the
    detector vector positions.

    Use the `positions` argument to change the vectors used as pixel positions.
    Sliders are added to navigate extra dimensions.
    Spatial slicing and pixel opacity control is available using the controls
    below the scene.
    Use the `pixel_size` argument to specify the size of the detectors.
    If no `pixel_size` is given, a guess is performed based on the distance
    between the positions of the first two pixel positions.
    The aspect ratio of the positions is preserved by default, but this can
    be changed to automatic scaling using `aspect="equal"`.

    `components` dictionary uses the key as the name to display the component.
    This can be any desired name, it does not have to relate to the input
    `scipp_obj` naming.
    The value for each entry is itself a dictionary that provides the display
    settings and requires:

    * `center` - scipp scalar vector describing position to place item at.
    * `size` - scipp scalar vector describing the bounding box to use in the
    same length units as positions
    * `type` - known shape type to use.
    Valid types are: 'box', 'cylinder' or 'disk'.

    Optional arguments are:

    * `color` - a hexadecimal color code such as #F00000 to use as fill or
    line color
    * `wireframe` - wireframe is a bool that defaults to False. If set to True,
    the returned geometrical shape is a wireframe instead of a shape with
    opaque faces

    Parameters
    ----------
    scipp_obj:
        Scipp object holding geometries.
    positions:
        Key for coord/attr holding positions to use for pixels.
    pixel_size:
        Custom pixel size to use for detector pixels.
    components:
        Dictionary containing display names and corresponding settings
        (also a Dictionary) for additional components to display.
    kwargs:
        Additional keyword arguments to pass to :func:`plopp.scatter3d`.

    Returns
    -------
    :
        The 3D plot object
    """
    if not p3:
        raise _pythreejs_import_error

    import plopp as pp

    positions_var = get_meta(scipp_obj)[positions]
    if pixel_size is None:
        pos_array = positions_var.values
        if len(pos_array) > 1:
            pixel_size = np.linalg.norm(pos_array[1] - pos_array[0])

    fig = pp.scatter3d(scipp_obj, pos=positions, pixel_size=pixel_size, **kwargs)
    scene = fig.canvas.scene

    # Add additional components from the beamline
    if components:
        _plot_components(scipp_obj, components, positions_var, scene)

    return fig
