File loading
============

Scippneutron is able to load Nexus files from ESS experimental runs,
as well as data files from other neutron facilities.
There are currently two different implementations of file loading.

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


``scippneutron.load_nexus``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Deprecated!** Use `ScippNexus <https://scipp.github.io/scippnexus/>`_ instead.

This was added to Scippneutron to load ESS Nexus files (or any file that
complies with the Nexus standard) without having to rely on Mantid.

It is written in Python, and uses the `h5py <https://www.h5py.org/>`_ library
to read the files.

Instead of looking for hard-coded path names, as is often done in the Mantid
loaders, it simply traverses the file tree and looks for any NXClass it can
read.

It can load data from both a Nexus file (``load_nexus``) or from a JSON string
(``load_nexus_json``), the latter being used when reading from a Kafka stream.
