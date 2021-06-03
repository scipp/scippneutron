.. _release-notes:

Release Notes
=============

Since v0.2.0
------------

Features
~~~~~~~~

Breaking changes
~~~~~~~~~~~~~~~~

Contributors
~~~~~~~~~~~~

Owen Arnold,
Simon Heybrock,
Matthew D. Jones,
Neil Vaytet,
and Jan-Lukas Wynen

v0.2.0 (June 2021)
-------------------

Features
~~~~~~~~

* ``convert`` new returns data arrays with a new coordinate array (for the converted dimension), but data and unrelated meta data is not deep-copied.
  This should improve performance in a number of cases.

Bugfixes
~~~~~~~~

* When converting from data from `Mantid <https://www.mantidproject.org/Main_Page>`_ with its `instrument <https://docs.mantidproject.org/nightly/concepts/InstrumentDefinitionFile.html>`_ format;
  Duplicate named detectors (including monitors) will have unique names created by concatenating the name with the spectrum number for that detector.
  This fixes a bug with monitors where previously, duplicate entries encoutered after the first were rejected from the output metadata.
  In the case of instruments such as POLARIS, all monitors will now be translated.

Contributors
~~~~~~~~~~~~

Owen Arnold,
Simon Heybrock,
Matthew D. Jones,
Neil Vaytet,
and Jan-Lukas Wynen

v0.1.0 (March 2021)
-------------------

Features
~~~~~~~~

* Functionality from ``scipp.neutron`` (as previously known as part of the scipp package) is now available in this package.
  This includes in particular the instrument view and "unit conversions" for time-of-flight neutron sources.
* Convert supports a greatly enhanced way of obtaining required parameters of the beamline.
  Instead of requiring raw component positions it can now work directly with, e.g., `two_theta`.
* Add scipp ``datetime64`` support in mantid-scipp converters `#39 <https://github.com/scipp/scipp/pull/39>`_.

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
