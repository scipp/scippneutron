.. _release-notes:

Release Notes
=============

First release (planned)
-----------------------

Features
~~~~~~~~

* Functionality from ``scipp.neutron`` (as previously known as part of the scipp package) is now available in this package.
  This includes in particular the instrument view and "unit conversions" for time-of-flight neutron sources.

Breaking changes
~~~~~~~~~~~~~~~~

* ``scipp.neutron.diffraction`` is NOT available in ``scippneutron`` since its original content is facility-specific and does not comply with the inclusion guidelines in this librarary.
* Naming convention for (in particular) coords and attrs used by unit conversion has changed.
  Generally what previously used hyphens `-` now uses underscore `_`.

  * ``pulse-time`` is now ``pulse_time``
  * ``sample-position`` is now ``sample_position``
  * ``source-position`` is now ``source_position``
  * ``energy-transfer`` is now ``energy_transfer``
  * ``incident-energy`` is now ``incident_energy``
  * ``final-energy`` is now ``final_energy``
  * ``d-spacing`` is now ``dspacing`` (no hyphen)

* ``convert`` now requires a mandatory argument ``scatter=True`` or ``scatter=False``.
  Previously the conversion mode was determined automatically based on the presence of a ``sample_position`` coordinate.
  This is error prone hidden/implicit behavior, which is now avoided.

Contributors
~~~~~~~~~~~~

Everyone contributing originally to ``scipp.neutron``.
