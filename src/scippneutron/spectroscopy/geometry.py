# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Tools for geometry calculations and representations."""

from __future__ import annotations

from dataclasses import dataclass

import scipp as sc


@dataclass(slots=True, kw_only=True, frozen=True)
class Plane:
    r"""A plane in 3 dimensions defined by two in-plane vectors and a normal vector.

    The vectors form a right-handed system with

    .. math::

        \vec{v}_1 \cross \vec{v}_2 = \vec{n}

    All vectors are automatically normalized and are thus dimensionless.
    """

    vector1: sc.Variable
    """First in-plane vector."""
    vector2: sc.Variable
    """Second in-plane vector."""
    normal: sc.Variable
    """Normal vector."""

    @classmethod
    def from_normal_and_reference(
        cls, *, normal: sc.Variable, reference: sc.Variable
    ) -> Plane:
        r"""Construct a plane from a normal vector and an in-plane vector.

        The plane vectors are computed as follows to produce an orthogonal system.

        .. math::

            \hat{v}_2 &= \hat{n} \cross \hat{r}_1 \\
            \hat{v}_1 &= \hat{v}_2 \cross \hat{n}

        All vectors are normalized.
        :math:`\hat{r}_i` are the normalized input reference vectors.

        This algorithm uses the direction of ``reference`` for ``vector``
         if it is orthogonal to ``normal``. But it computes a new vector otherwise.

        Parameters
        ----------
        normal:
            Normal vector of the plane.
        reference:
            An in-plane vector.

        Returns
        -------
        :
            A plane with ``vector2 = cross(normal, vector1)``.
        """
        vector2 = sc.cross(normal, reference)
        vector1 = sc.cross(vector2, normal)
        return cls(
            vector1=vector1 / sc.norm(vector1),
            vector2=vector2 / sc.norm(vector2),
            normal=normal / sc.norm(normal),
        )

    @classmethod
    def from_in_plane_vectors(
        cls, reference1: sc.Variable, reference2: sc.Variable
    ) -> Plane:
        r"""Construct a plane from two in-plane vectors.

        The plane vectors are computed as follows to produce an orthogonal system.

        .. math::

            \hat{v}_1 &= \hat{r}_1 \\
            \hat{n} &= \hat{v}_1 \cross \hat{r}_2 \\
            \hat{v}_2 &= \hat{n} \cross \hat{v}_1

        All vectors are normalized.
        :math:`\hat{r}_i` are the normalized input reference vectors.
        Note that ``reference1`` is used as the direction of ``vector1`` while
        ``reference2`` and ``vector2`` are not necessarily parallel.

        Parameters
        ----------
        reference1:
            First in-plane vector.
        reference2:
            Second in-plane vector.

        Returns
        -------
        :
            A plane with ``normal = cross(vector1, vector2)``.
        """
        normal = sc.cross(reference1, reference2)
        vector2 = sc.cross(normal, reference1)
        return cls(
            vector1=reference1 / sc.norm(reference1),
            vector2=vector2 / sc.norm(vector2),
            normal=normal / sc.norm(normal),
        )
