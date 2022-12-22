.. _release-notes:

Release Notes
=============


.. Template, copy this to create a new section after a release:

   v0.xy.0 (Unreleased)
   --------------------

   Features
   ~~~~~~~~

   Breaking changes
   ~~~~~~~~~~~~~~~~

   Bugfixes
   ~~~~~~~~

   Documentation
   ~~~~~~~~~~~~~

   Deprecations
   ~~~~~~~~~~~~

   Stability, Maintainability, and Testing
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   Contributors
   ~~~~~~~~~~~~

   Simon Heybrock :sup:`a`\ ,
   Neil Vaytet :sup:`a`\ ,
   and Jan-Lukas Wynen :sup:`a`

v22.12.4 (December 2022)
------------------------

Bugfixes
~~~~~~~~

* :func:`scippneutron.instrument_view` now works again when ``plopp`` is enabled for Scipp plotting.
* :func:`scippneutron.load_nexus` fixed for cases of monitors or detectors without data, which led to aborting the load with scippnexus-22.12.3, instead of continuing.

v22.12.0 (December 2022)
------------------------

Features
~~~~~~~~

* ScippNeutron is now available on PyPI `#384 <https://github.com/scipp/scipp/pull/384>`_.

Breaking changes
~~~~~~~~~~~~~~~~

* Removed C++ components and CMake package configuration. This only affects users that linked against the C++ part of ScippNeutron `#384 <https://github.com/scipp/scipp/pull/384>`_.

Stability, Maintainability, and Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ScippNeutron no longer depends on a specific version of Scipp`#384 <https://github.com/scipp/scipp/pull/384>`_.

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.11.0 (November 2022)
-----------------------

Features
~~~~~~~~

* The instrument view will now use ``plopp`` if it is the current default for plotting `#378 <https://github.com/scipp/scipp/pull/378>`_.

Breaking changes
~~~~~~~~~~~~~~~~

* The instrument view no longer accepts a dataset as input, only data arrays are supported `#378 <https://github.com/scipp/scipp/pull/378>`_.

Bugfixes
~~~~~~~~

* :func:`scippneutron.load_nexus` now works with scippnexus 0.4.1 `#380 <https://github.com/scipp/scipp/pull/380>`_.

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.10.0 (September 2022)
------------------------

Breaking changes
~~~~~~~~~~~~~~~~

* Switched to scipp 0.17 and remove pin on patch version

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.9.4 (September 2022)
-----------------------

Breaking changes
~~~~~~~~~~~~~~~~

* Switched to scipp 0.16.4

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.9.3 (September 2022)
-----------------------

Features
~~~~~~~~

* Update for compatibility with scippnexus v0.3 `#370 <https://github.com/scipp/scipp/pull/370>`_.


v0.9.2 (August 2022)
--------------------

Features
~~~~~~~~

* Made compatible with scippnexus v0.2 `#369 <https://github.com/scipp/scipp/pull/369>`_.

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.9.1 (August 2022)
--------------------

Breaking changes
~~~~~~~~~~~~~~~~

* Switched to scipp 0.16.2

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.9.0 (August 2022)
--------------------

Features
~~~~~~~~

* Kernels for coordinate transformations are now public in :mod:`scippneutron.conversion` `#361 <https://github.com/scipp/scipp/pull/361>`_.

Bugfixes
~~~~~~~~

* Fixed ``dspacing_from_wavelength``, results used to be wrong by a factor of ``10**10`` `#361 <https://github.com/scipp/scipp/pull/361>`_.
* ``two_theta`` as used by coordinate transformations now uses a numerically more stable implementation, the old one had an error of up to ``10**-6`` for small angles `#361 <https://github.com/scipp/scipp/pull/361>`_.

Breaking changes
~~~~~~~~~~~~~~~~

* ``scippneutron.tof.conversions`` has been split into :mod:`scippneutron.conversion.graph.beamline` and :mod:`scippneutron.conversion.graph.tof` `#361 <https://github.com/scipp/scipp/pull/361>`_.
* Switched to scipp 0.16.1

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.8.0 (July 2022)
------------------

Breaking changes
~~~~~~~~~~~~~~~~

* Switched to scipp 0.15.

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.7.0 (June 2022)
------------------

Breaking changes
~~~~~~~~~~~~~~~~

* Switched to scipp 0.14.

Features
~~~~~~~~

* Started releasing for Apple arm64 architecture.

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.6.0 (May 2022)
-----------------

Breaking changes
~~~~~~~~~~~~~~~~

* Remove accidental dependency on Mantid. Users now have to install Mantid themselves if they need it `#332 <https://github.com/scipp/scipp/pull/332>`_.
* Removed module ``scippneutron.nexus`` in favor of `scippnexus <https://scipp.github.io/scippnexus/>`_ to implement :func:`scippneutron.load_nexus`.

Bugfixes
~~~~~~~~

* Fixed loading event data for monitors that is stored in a separate NeXus group.

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.5.2 (March 2022)
-------------------

Breaking changes
~~~~~~~~~~~~~~~~

* Some potentially breaking changes in :py:func`scippneutron.load_nexus`.

Bugfixes
~~~~~~~~

* Fixed resource leak in data streaming `#298 <https://github.com/scipp/scippneutron/pull/298>`_.

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
Tom Willemsen :sup:`b, c`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.5.0 (February 2022)
----------------------

Features
~~~~~~~~

* Added Utilities for unwrapping frames `#242 <https://github.com/scipp/scippneutron/pull/242>`_.
* Added First draft of low-level utilities for loading NeXus files `#249 <https://github.com/scipp/scippneutron/pull/249>`_.
* Transformation chains containing multiple values (based on ``NXlog`` groups) are now loaded by :func:`scippneutron.load_nexus` `#267 <https://github.com/scipp/scippneutron/pull/267>`_.

Bugfixes
~~~~~~~~

* Fixed bug in ``load_nexus``, which interpreted ``NXtransformations`` as passive transformations `#275 <https://github.com/scipp/scippneutron/pull/275>`_.

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
