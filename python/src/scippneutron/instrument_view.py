# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet and Owen Arnold

import numpy as np
import pythreejs as p3
import scipp as sc
from scipy.spatial.transform import Rotation as Rot


def _text_mesh(position, display_text, x_width, y_width):
    text_geometry = p3.PlaneGeometry(width=x_width,
                                     height=y_width,
                                     widthSegments=2,
                                     heightSegments=2)

    text = p3.TextTexture(string=display_text, color='black', size=20)
    text_material = p3.MeshBasicMaterial(map=text, transparent=True)
    text_mesh = p3.Mesh(geometry=text_geometry, material=text_material)
    text_mesh.position = position
    return text_mesh


def _box(position, display_text, bounding_box, color, **kwargs):
    geometry = p3.BoxGeometry(width=bounding_box[0],
                              height=bounding_box[1],
                              depth=bounding_box[2],
                              widthSegments=2,
                              heightSegments=2,
                              depthSegments=2)

    material = p3.MeshBasicMaterial(color=color)
    mesh = p3.Mesh(geometry=geometry, material=material)
    mesh.position = tuple(position.value)
    text_position = tuple(position.value + np.array([0, 0.7 * bounding_box[1], 0]))
    text_mesh = _text_mesh(text_position,
                           display_text=display_text,
                           x_width=max(bounding_box),
                           y_width=max(bounding_box))
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


def _disk_chopper(position, display_text, bounding_box, color, **kwargs):
    geometry = p3.CylinderGeometry(radiusTop=bounding_box[0] / 2,
                                   radiusBottom=bounding_box[0] / 2,
                                   height=bounding_box[0] / 100,
                                   radialSegments=24,
                                   heightSegments=12,
                                   openEnded=False,
                                   thetaStart=np.pi / 8,
                                   thetaLength=2 * np.pi - (np.pi / 8))

    material = p3.MeshBasicMaterial(color=color)
    mesh = p3.Mesh(geometry=geometry, material=material)
    mesh.position = tuple(position.value)
    beam = _find_beam(det_com=kwargs['det_center'], pos=position)
    disk_axis = np.array([0, 1, 0])  # Disk created with this axis
    rotation = _alignment_matrix(to_align=disk_axis, target=beam)
    mesh.setRotationFromMatrix(rotation.flatten())

    text_position = tuple(position.value + np.array([0, 0.7 * bounding_box[1], 0]))
    text_mesh = _text_mesh(text_position,
                           display_text=display_text,
                           x_width=max(bounding_box),
                           y_width=max(bounding_box))
    return mesh, text_mesh


def _cylinder(position, display_text, bounding_box, color, **kwargs):
    geometry = p3.CylinderGeometry(radiusTop=bounding_box[0] / 2,
                                   radiusBottom=bounding_box[0] / 2,
                                   height=bounding_box[1],
                                   radialSegments=12,
                                   heightSegments=12,
                                   openEnded=False,
                                   thetaStart=0,
                                   thetaLength=2.0 * np.pi)

    material = p3.MeshBasicMaterial(color=color)
    mesh = p3.Mesh(geometry=geometry, material=material)
    mesh.position = tuple(position.value)
    # Position label above cylinder
    text_position = tuple(position.value + np.array([0, 0.7 * bounding_box[1], 0]))
    text_mesh = _text_mesh(text_position,
                           display_text=display_text,
                           x_width=max(bounding_box),
                           y_width=max(bounding_box))
    return mesh, text_mesh


def _unpack_to_scene(scene, items):
    if hasattr(items, "__iter__"):
        for item in items:
            scene.add(item)
    else:
        scene.add(items)


def _add_to_scene(position, scene, shape, display_text, bounding_box, color, **kwargs):
    _unpack_to_scene(
        scene,
        shape(position,
              display_text=display_text,
              bounding_box=bounding_box,
              color=color,
              **kwargs))


def _furthest_component(det_center, scipp_obj, additional):
    distances = [(settings["at"],
                  sc.norm(scipp_obj.meta[settings["at"]] - det_center).value)
                 for settings in list(additional.values())]
    item, max_displacement = sorted(distances, key=lambda x: x[1])[-1]
    return item, max_displacement


def _plot_components(scipp_obj, components, positions_var, scene):
    det_center = sc.mean(positions_var)
    furthest_key, furthest_distance = _furthest_component(det_center, scipp_obj,
                                                          components)
    # Some scaling to set width according to distance from detector center
    shapes = {"box": _box, "cylinder": _cylinder, "disk": _disk_chopper}
    for name, settings in components.items():
        type = settings["type"]
        size = settings["size"]
        at = settings["at"]
        color = settings.get("color", "#808080")
        if type not in shapes:
            supported_shapes = ", ".join(shapes.keys())
            raise ValueError(f"Unknown shape: {type} requested for {name}. "
                             f"Allowed values are: {supported_shapes}")
        component_position = scipp_obj.meta[at]
        _add_to_scene(position=component_position,
                      scene=scene,
                      shape=shapes[type],
                      display_text=name,
                      bounding_box=tuple(size),
                      color=color,
                      det_center=det_center)
    # Reset camera
    camera = _get_camera(scene)
    if camera:
        camera.far = max(camera.far, furthest_distance * 5.0)


def _get_camera(scene):
    for child in scene.children:
        if isinstance(child, p3.PerspectiveCamera):
            return child
    return None


def instrument_view(scipp_obj=None,
                    positions="position",
                    pixel_size=None,
                    components=None,
                    **kwargs):
    """
    :param scipp_obj: scipp object holding geometries
    :param positions: Key for coord/attr holding positions to use for pixels
    :param pixel_size: Custom pixel size to use for detector pixels
    :param components: Dictionary containing display names and corresponding
    settings (also a Dictionary) for additional components to display
     items with known positions to be shown
    :param kwargs: Additional keyword arguments to pass to scipp.plotting.plot
    :return: The 3D plot object

    Plot a 3D view of the instrument, using the `position` coordinate as the
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
    `scipp_obj` naming. The value for each entry is itself a dictionary that
    provides the display settings and requires an `at`, `size`, `type` and
    optionally a `color`. `at` should correspond to the meta item where the
    position to place the component is taken from. `size` is a 3-element tuple
    describing the bounding box to use. `type` is one of any known shape types
    including box, cylinder and disk. The `color` is a hexidecimal color code
    such as #F00000 to use as fill color.
    """

    from scipp.plotting import plot
    from scipp.plotting.objects import PlotDict

    positions_var = scipp_obj.meta[positions]
    if pixel_size is None:
        pos_array = positions_var.values
        if len(pos_array) > 1:
            pixel_size = np.linalg.norm(pos_array[1] - pos_array[0])

    plt = plot(scipp_obj,
               projection="3d",
               positions=positions,
               pixel_size=pixel_size,
               **kwargs)

    # Add additional components from the beamline
    if components and not isinstance(plt, PlotDict):
        scene = plt.view.figure.scene
        _plot_components(scipp_obj, components, positions_var, scene)

    return plt
