File loading
============

Scippneutron does not provide routines to read files itself.
However, it is able to load data files from many neutron facilities by using Mantid.
For NeXus files, prefer `ScippNexus <https://scipp.github.io/scippnexus/>`_
over Mantid.

``scippneutron.load_with_mantid``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This classic (or legacy) file loading method wraps
the `Load <https://docs.mantidproject.org/nightly/algorithms/Load-v1.html>`_
algorithm from the Mantid framework to load the files and convert from a
Mantid workspace to a Scipp data array.

The Mantid file loaders are written in C++.

This is the method that allows Scippneutron users to load many different
file formats from a multitude of existing facilities.

For more details, see
`scippneutron.load_with_mantid <../generated/functions/scippneutron.load_with_mantid.rst>`_.
