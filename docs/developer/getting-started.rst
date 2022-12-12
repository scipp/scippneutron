Getting Started
===============

What goes into ScippNeutron
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Code contributions to ScippNeutron should meet the following criteria:

* Relate to neutron scattering data reduction in general or be particular to a technique
* Will *not* contain any instrument or facility specific assumptions or branching
* Will provide unit tests, which demonstrate a fixed regression or exercise a new feature
  - Unit test suites will execute rapidly, no slow tests
  - As far as possible, unit test will make no use of external data via IO

In addition, all code should meet the code :ref:`conventions`.

Setting up
~~~~~~~~~~

Getting the code
^^^^^^^^^^^^^^^^

Clone the git repository from `GitHub <https://github.com/scipp/scippneutron>`_.

Dependencies
^^^^^^^^^^^^

ScippNeutron can be installed either using pip or conda.
Since Mantid is only available on conda, it is recommended to use that over pip.

Conda
"""""

There is no predefined environment file with dependencies as each current developer has their own file with a custom collection of development dependencies.
At the least, all ``run`` and ``test`` dependencies specified in ``conda/meta.yaml`` as well as ``conda-build`` are required.

Pip
"""

Use ``pip install -r requirements/dev.txt`` to install dependencies.
The package versions match those used in CI.

Installing ScippNeutron locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conda
"""""

Use ``conda develop src`` to make ScippNeutron importable from source.

Pip
"""

Use ``python -m pip install -e .`` to install in editable mode.

Tutorial and Test Data
~~~~~~~~~~~~~~~~~~~~~~

There are a number of data files which can be downloaded automatically by scippneutron.
The functions in `scippneutron.data` download and cache these files if and when they are used.
By default, the files are stored in the OS's cache directory.
The location can be customized by setting the environment variable ``SCIPPNEUTRON_DATA_DIR``
to the desired path.
