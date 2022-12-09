# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
from typing import Union

import scipp as sc

from .conversion import graph


def _derived_coord(da: Union[sc.DataArray, sc.Dataset],
                   name: str,
                   scatter: bool = True) -> sc.Variable:
    tmp = da.transform_coords(
        name,
        graph=graph.beamline.beamline(scatter=scatter),
        rename_dims=False,
        keep_aliases=False,
        keep_inputs=False,
        keep_intermediate=False,
    )
    return tmp.coords[name]


def position(da: Union[sc.DataArray, sc.Dataset]) -> sc.Variable:
    """Extract the detector pixel positions from a data array or dataset.

    Parameters
    ----------
    da:
        Get or compute the positions from coords and attrs of this.

    Returns
    -------
    :
        The detector pixel positions.
    """
    return _derived_coord(da, 'position')


def source_position(da: Union[sc.DataArray, sc.Dataset]) -> sc.Variable:
    """Extract the position of the neutron source from a data array or dataset.

    Parameters
    ----------
    da:
        Get or compute the source position from coords and attrs of this.

    Returns
    -------
    :
        The source position.
    """
    return _derived_coord(da, 'source_position')


def sample_position(da: Union[sc.DataArray, sc.Dataset]) -> sc.Variable:
    """Extract the position of the sample from a data array or dataset.

    Parameters
    ----------
    da:
        Get or compute the sample position from coords and attrs of this.

    Returns
    -------
    :
        The sample position.
    """
    return _derived_coord(da, 'sample_position')


def incident_beam(da: Union[sc.DataArray, sc.Dataset]) -> sc.Variable:
    """Extract the incident beam vector from a data array or dataset.

    This is the direction and length of the primary flight path,
    i.e. from source to sample.

    Parameters
    ----------
    da:
        Get or compute the incident beam from coords and attrs of this.

    Returns
    -------
    :
        The incident beam.
    """
    return _derived_coord(da, 'incident_beam')


def scattered_beam(da: Union[sc.DataArray, sc.Dataset]) -> sc.Variable:
    """Extract the scattered beam vector from a data array or dataset.

    This is the direction and length of the secondary flight path,
    i.e. from sample to detector.

    Parameters
    ----------
    da:
        Get or compute the scattered beam from coords and attrs of this.

    Returns
    -------
    :
        The scattered beam.
    """
    return _derived_coord(da, 'scattered_beam')


def Ltotal(da: Union[sc.DataArray, sc.Dataset], scatter: bool) -> sc.Variable:
    """Extract the length of the total flight path from a data array or dataset.

    Parameters
    ----------
    da:
        Get or compute the total flight path length from coords and attrs of this.
    scatter:
        If ``True``, assume a beam that scattered off a sample and return the
        sum of L1 and L2.
        If ``False``, return the straight distance between source and detector.

    Returns
    -------
    :
        The length of the total flight path.
    """
    return _derived_coord(da, 'Ltotal', scatter=scatter)


def L1(da: Union[sc.DataArray, sc.Dataset]) -> sc.Variable:
    """Extract the length of the primary flight path from a data array or dataset.

    This is the distance between neutron source and sample.

    Parameters
    ----------
    da:
        Get or compute the primary flight path length from coords and attrs of this.

    Returns
    -------
    :
        The length of the primary flight path.
    """
    return _derived_coord(da, 'L1')


def L2(da: Union[sc.DataArray, sc.Dataset]) -> sc.Variable:
    """Extract the length of the secondary flight path from a data array or dataset.

    This is the distance between sample and detector.

    Parameters
    ----------
    da:
        Get or compute the secondary flight path length from coords and attrs of this.

    Returns
    -------
    :
        The length of the secondary flight path.
    """
    return _derived_coord(da, 'L2')


def two_theta(da: Union[sc.DataArray, sc.Dataset]) -> sc.Variable:
    """Extract the scattering angle from a data array or dataset.

    The angle is defined as in Bragg's law.
    See the beamline `documentation <../modules/scippneutron.conversion.beamline.rst>`_
    for a full definition.

    Parameters
    ----------
    da:
        Get or compute the scattering angle from coords and attrs of this.

    Returns
    -------
    :
        Twice the scattering angle.
    """
    return _derived_coord(da, 'two_theta')
