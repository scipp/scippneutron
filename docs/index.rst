ScippNeutron - Neutron scattering data processing based on Scipp
================================================================

Toolkit for neutron scattering data reduction powered by `scipp <https://scipp.github.io/>`_.
Provides "unit conversions" and technique specific tools based on physics of (time-of-flight) neutron scattering.

News
----

- [|SCIPPNEUTRON_RELEASE_MONTH|] scippneutron-|SCIPPNEUTRON_VERSION| `has been released <about/release-notes.rst>`_.
  Check out the `What's new <about/whats-new.rst>`_ notebook for an overview of recent highlights and major changes.
- [December 2022] ScippNeutron is now also available on PyPI.
  See the updated ``pip`` `installation instructions <getting-started/installation.rst#Pip>`_.

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

   user-guide/coordinate-transformations
   user-guide/groupby
   user-guide/from-mantid-to-scipp
   user-guide/instrument-view
   user-guide/recipes

.. toctree::
   :caption: Reference
   :maxdepth: 3

   reference/free-functions
   reference/modules
   reference/frame-unwrapping

.. toctree::
   :caption: Tutorials
   :maxdepth: 3

   tutorials/1_exploring-data
   tutorials/2_working-with-masks
   tutorials/3_understanding-event-data
   tutorials/powder-diffraction

.. toctree::
   :caption: Developer Documentation
   :maxdepth: 2

   developer/getting-started
   developer/coding-conventions
   developer/data-stream
   developer/file-loading

.. toctree::
   :caption: About
   :maxdepth: 3

   about/about
   about/whats-new
   about/contributing
   about/release-notes
