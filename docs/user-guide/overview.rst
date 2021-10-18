.. _scipp-neutron:
.. currentmodule:: scippneutron

Overview: scippneutron
======================

``scippneutron`` builds on top of the core ``scipp`` package and provides features specific to handling data generated in neutron scattering facilities.
A key example is "unit conversion", e.g., from time-of-flight to energy transfer in an inelastic neutron scattering experiment at a spallation-based neutron source.

Free functions
--------------

Mantid Compatibility
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../generated

   from_mantid
   to_mantid
   fit

Unit Conversion
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../generated

   convert
   conversion_graph
   deduce_conversion_graph

Beamline geometry
~~~~~~~~~~~~~~~~~

Note that ``theta`` or ``scattering_angle`` are deliberately not supported, due to some ambiguity on how the terms are used in the community, and possible confusion of ``theta`` (from Bagg's law) with ``theta`` in spherical coordinates.

.. autosummary::
   :toctree: ../generated

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
   :toctree: ../generated

   load
   load_nexus
