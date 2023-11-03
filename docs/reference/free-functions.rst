.. currentmodule:: scippneutron

Free functions
==============

Mantid Compatibility
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../generated/functions

   array_from_mantid
   from_mantid
   load_with_mantid
   to_mantid
   fit

Coordinate transformations (Unit conversion)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../generated/functions

   convert

Beamline geometry
~~~~~~~~~~~~~~~~~

Note that ``theta`` or ``scattering_angle`` are deliberately not supported, due to some ambiguity on how the terms are used in the community, and possible confusion of ``theta`` (from Bagg's law) with ``theta`` in spherical coordinates.

.. autosummary::
   :toctree: ../generated/functions

   position
   source_position
   sample_position
   Ltotal
   L1
   L2
   two_theta
   incident_beam
   scattered_beam

Loading Nexus files
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../generated/functions

   load_nexus
