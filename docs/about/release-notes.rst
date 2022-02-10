.. _release-notes:

Release Notes
=============

v0.5.0 (unreleased)
--------------------

Features
~~~~~~~~

Breaking changes
~~~~~~~~~~~~~~~~

Bugfixes
~~~~~~~~

* Fix bug in ``load_nexus``, which interpreted ``NXtransformations`` as passive transformations `#275 <https://github.com/scipp/scippneutron/pull/275>`_.
* Fix bug in ``load_nexus``, ignoring ``offset`` attribute of rotation in ``NXtransformations`` `#275 <https://github.com/scipp/scippneutron/pull/275>`_.
* Fix bug in ``load_nexus``, misinterpreting ``offset`` attribute of rotation in ``NXtransformations`` `#275 <https://github.com/scipp/scippneutron/pull/275>`_.

Deprecations
~~~~~~~~~~~~

Stability, Maintainability, and Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
Tom Willemsen :sup:`b, c`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.4.2 (January 2022)
---------------------

Bugfixes
~~~~~~~~

* Change output dtype of graphs for coordinate transformations to always be floating point, fixes incorrect truncation of the result to integer if, e.g. ``tof`` is an integer (this also affects ``convert``) `#230 <https://github.com/scipp/scippneutron/pull/230>`_.
* Fix bug in ``load_nexus`` which prevented nexus files containing any empty datasets from being loaded correctly.

v0.4.1 (November 2021)
----------------------

Bugfixes
~~~~~~~~

* Fix bug in ``load`` that loaded weighted events without their weights if the first spectrum is empty `#211 <https://github.com/scipp/scippneutron/pull/211>`_.

v0.4.0 (October 2021)
---------------------

Features
~~~~~~~~

* Add ``tof.conversions`` module with building blocks for custom coordinate transformation graphs `#187 <https://github.com/scipp/scipp/pull/187>`_.

Breaking changes
~~~~~~~~~~~~~~~~

* Changed behavior of ``convert`` `#162 <https://github.com/scipp/scipp/pull/162>`_.

  * It is no longer possible to convert *to* time-of-flight.
  * To compensate, it is now possible to convert between wavelength, energy, and d-spacing directly.
  * Some input coords which used to be preserved are now turned into attributes.
    See `Coordinate transformations <https://scipp.github.io/user-guide/coordinate-transformations.html>`_ in scipp for details.
  * The ``out`` argument is no longer supported.

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Samuel Jones :sup:`b`\ ,
Neil Vaytet :sup:`a`\ ,
Tom Willemsen :sup:`b, c`\ ,
and Jan-Lukas Wynen :sup:`a`\

v0.3.0 (September 2021)
-----------------------

Features
~~~~~~~~

* ``load_nexus`` will read ub_matrix and orientation_matrix information from nexus files. Likewise, the Mantid converters will propagate the same information if present.
* ``load_nexus`` now has an optional flag, ``raw_detector_data``, which specifies that detector and event data should be loaded as it appears in the nexus file (without any binning or preprocessing).
* ``load_nexus`` will now load monitor data from nexus files.
* ``load_nexus`` will now load pulse times along with event data.
* ``instrument_view`` can display extra beamline components.

Breaking changes
~~~~~~~~~~~~~~~~

* ``load_nexus`` will now add a single TOF bin around event data

Contributors
~~~~~~~~~~~~

Owen Arnold :sup:`b, c`\ ,
Simon Heybrock :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
Tom Willemsen :sup:`b, c`\ ,
and Jan-Lukas Wynen :sup:`a`\

v0.2.0 (June 2021)
-------------------

Features
~~~~~~~~

* ``convert`` new returns data arrays with a new coordinate array (for the converted dimension), but data and unrelated meta data is not deep-copied.
  This should improve performance in a number of cases.
* ``load_nexus`` will read in chopper positions and frequencies if written as ``NXdisk_choppers`` (see NeXus format) from the file
* ``instrument_view`` can show the positions of non-detector components such as choppers, and the sample on the beamline.

Bugfixes
~~~~~~~~

* When converting from data from `Mantid <https://www.mantidproject.org/Main_Page>`_ with its `instrument <https://docs.mantidproject.org/nightly/concepts/InstrumentDefinitionFile.html>`_ format;
  Duplicate named detectors (including monitors) will have unique names created by concatenating the name with the spectrum number for that detector.
  This fixes a bug with monitors where previously, duplicate entries encoutered after the first were rejected from the output metadata.
  In the case of instruments such as POLARIS, all monitors will now be translated.
* ``load_nexus`` will no longer fail to load nexus files containing strings with non-ascii characters, for example a log with units of '°'.

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

Contributing Organizations
--------------------------
* :sup:`a`\  `European Spallation Source ERIC <https://europeanspallationsource.se/>`_, Sweden
* :sup:`b`\  `Science and Technology Facilities Council <https://www.ukri.org/councils/stfc/>`_, UK
* :sup:`c`\  `Tessella <https://www.tessella.com/>`_, UK
