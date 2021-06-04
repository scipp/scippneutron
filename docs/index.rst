scippneutron - Neutron scattering data processing based on scipp
================================================================

Toolkit for neutron scattering data reduction powered by `scipp <https://scipp.github.io/>`_.
Provides "unit conversions" and technique specific tools based on physics of (time-of-flight) neutron scattering.

News
----

- Just like scipp, scippneutron has switched from GPLv3 to the more permissive BSD-3 license which fits better into the Python eco system.
- [June 2021] `scippneutron-0.2 <https://scipp.github.io/scippneutron/about/release-notes.html#v0-2-0-june-2021>`_ has been released.
- [June 2021] `scipp-0.7 <https://scipp.github.io/about/release-notes.html#v0-7-0-june-2021>`_ has been released.
- [March 2021] `scippneutron-0.1 <https://scipp.github.io/scippneutron/about/release-notes.html#v0-1-0-march-2021>`_ has been released.
- [March 2021] `scipp-0.6 <https://scipp.github.io/about/release-notes.html#v0-6-0-march-2021>`_ has been released.
  The `What's new <https://scipp.github.io/about/whats-new/whats-new-0.6.0.html>`_ notebook provides an overview of the highlights and major changes in both scipp and scippneutron.

Where can I get help?
---------------------

For questions not answered in the documentation
`this page <https://github.com/scipp/scippneutron/issues?utf8=%E2%9C%93&q=label%3Aquestion>`_
provides a forum with discussions on problems already met/solved in the community.

New question can be asked by
`opening <https://github.com/scipp/scippneutron/issues/new?assignees=&labels=question&template=question.md&title=>`_
a new |QuestionLabel|_ issue.

.. |QuestionLabel| image:: images/question.png
.. _QuestionLabel: https://github.com/scipp/scippneutron/issues/new?assignees=&labels=question&template=question.md&title=

Documentation
=============

.. toctree::
   :caption: Getting Started
   :maxdepth: 3

   getting-started/installation

.. toctree::
   :caption: User Guide
   :maxdepth: 3

   user-guide/overview
   user-guide/unit-conversions
   user-guide/groupby
   user-guide/from-mantid-to-scipp
   user-guide/instrument-view

.. toctree::
   :caption: Tutorials
   :maxdepth: 3

   tutorials/exploring-data
   tutorials/working-with-masks
   tutorials/powder-diffraction

.. toctree::
   :caption: Developer Documentation
   :maxdepth: 2

   developer/getting-started
   developer/coding-conventions
   developer/testing-live-data
   developer/deployment

.. toctree::
   :caption: About
   :maxdepth: 3

   about/about
   about/contributing
   about/release-notes
