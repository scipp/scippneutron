# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

import numpy as np
import pythreejs as p3
import scipp as sc


def _cube(position, message, sz, color):
    geometry = p3.BoxGeometry(width=sz[0],
                              height=sz[1],
                              depth=sz[2],
                              widthSegments=2,
                              heightSegments=2,
                              depthSegments=2)

    material = p3.MeshBasicMaterial(color=color)
    mesh = p3.Mesh(geometry=geometry, material=material)
    mesh.position = position
    text_position = (position[0], position[1] + sz[1], position[2])
    text_mesh = _text_mesh(text_position, message, x_width=sz[0], y_width=sz[1])
    return mesh, text_mesh


def _text_mesh(position, message, x_width, y_width):
    text_geometry = p3.PlaneGeometry(width=x_width,
                                     height=y_width,
                                     widthSegments=2,
                                     heightSegments=2)

    text = p3.TextTexture(string=message, color='black', size=20)
    text_material = p3.MeshBasicMaterial(map=text, transparent=True)
    text_mesh = p3.Mesh(geometry=text_geometry, material=text_material)
    text_mesh.position = position
    return text_mesh


def _cylinder(position, message, sz, color):
    geometry = p3.CylinderGeometry(radiusTop=sz[0] / 2,
                                   radiusBottom=sz[0] / 2,
                                   height=sz[1],
                                   radialSegments=12,
                                   heightSegments=12,
                                   openEnded=False,
                                   thetaStart=0,
                                   thetaLength=2.0 * np.pi)

    material = p3.MeshBasicMaterial(color=color)
    mesh = p3.Mesh(geometry=geometry, material=material)
    mesh.position = position
    # Position label above cylinder
    text_position = (position[0], position[1] + sz[1], position[2])
    text_mesh = _text_mesh(text_position, message=message, x_width=sz[0], y_width=sz[1])
    return mesh, text_mesh


def _unpack_to_scene(scene, items):
    if hasattr(items, "__iter__"):
        for item in items:
            scene.add(item)
    else:
        scene.add(items)


def _add_source(position, scene, sz=(1, 1, 1)):
    _unpack_to_scene(scene, _cylinder(position,
                                      message="source",
                                      sz=sz,
                                      color="#808080"))


def _add_sample(position, scene, sz=(1, 1, 1)):
    _unpack_to_scene(scene, _cube(position, message="sample", sz=sz, color="#808080"))


def _calculate_component_width(det_center, component, fixed_width=None):
    if fixed_width:
        return fixed_width
    else:
        scaling_factor = 1 / 5.0
        return sc.norm(det_center - component).value * scaling_factor


def _add_beamline(scipp_obj, positions_var, scene):
    sample_width = None
    det_center = sc.mean(positions_var)
    beamline_comp_positions = []

    if "sample_position" in scipp_obj.meta:
        sample_width = _calculate_component_width(det_center,
                                                  scipp_obj.meta["sample_position"])
        sample = scipp_obj.meta["sample_position"]
        beamline_comp_positions.append(sc.norm(sample - det_center).value)
        _add_sample(tuple(sample.value), scene, sz=([sample_width] * 3))
    if "source_position" in scipp_obj.meta:
        source_width = _calculate_component_width(det_center,
                                                  scipp_obj.meta["source_position"],
                                                  sample_width)
        source = scipp_obj.meta["source_position"]
        beamline_comp_positions.append(sc.norm(source - det_center).value)
        _add_source(tuple(source.value), scene, sz=([source_width] * 3))

    if not beamline_comp_positions:
        return None
    return np.max(np.array(beamline_comp_positions)) * 5.0


def _get_camera(plt):
    for child in plt.view.figure.scene.children:
        if isinstance(child, p3.PerspectiveCamera):
            return child
    return None


def instrument_view(scipp_obj=None,
                    positions="position",
                    pixel_size=None,
                    plot_non_pixels=True,
                    **kwargs):
    """
    :param scipp_obj: scipp object holding geometries
    :param positions: Key for coord/attr holding positions to use for pixels
    :param pixel_size: Custom pixel size to use for detector pixels
    :param plot_non_pixels: If True render other known beamline components.
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
    """

    from scipp.plotting import plot

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
    # TODO solve hasattr fix
    if plot_non_pixels and hasattr(plot, 'view'):
        scene = plt.view.figure.scene
        z = _add_beamline(scipp_obj, positions_var, scene)
        # Reset camera
        if z:
            camera = _get_camera(plt)
            camera.far = z * 5.0

    return plt
