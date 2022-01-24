Getting Started
===============

What goes into scippneutron
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Code contributions to scippneutron should meet the following criteria:

* Relate to neutron scattering data reduction in general or be particular to a technique
* Will *not* contain any instrument or facility specific assumptions or branching
* Will provide unit tests, which demonstrate a fixed regression or exercise a new feature
  - Unit test suites will execute rapidly, no slow tests
  - As far as possible, unit test will make no use of external data via IO

In addition, all code should meet the code :ref:`conventions`.

Prerequisites
~~~~~~~~~~~~~

All non-optional build dependencies are installed automatically through Conan when running CMake.
Conan itself can be installed manually but we recommend using the generated ``scippneutron-developer.yml``
for installing this and other dependencies in a ``conda`` environment (see below).
Alternatively you can refer to this file for a full list of dependencies.

See `Tooling (scipp) <https://scipp.github.io/reference/developer/tooling.html>`_ for compilers and other required tools.

Getting the code, building, and installing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You first need to clone the git repository (either via SSH or HTTPS) from `GitHub <https://github.com/scipp/scippneutron>`_.
Note that this assumes you will end up with a directory structure similar to the following.
If you want something different be sure to modify paths as appropriate.

.. code-block::

  |-- scippneutron (source code)
  |   |-- build (build directory)
  |   |-- install (Python library installation)
  |   |-- ...
  |-- ...

To build and install the library:

.. code-block:: bash

  # Create Conda environment with dependencies and development tools
  python tools/metatoenv.py --dir=conda --env-file=scippneutron-developer.yml \
    --channels=conda-forge,scipp,ess-dmsc
  conda env create -f scippneutron-developer.yml
  conda activate scippneutron-developer

  # Create build and library install directories
  mkdir build
  mkdir install
  cd build

To build a debug version of the library:

.. code-block:: bash

  cmake \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPython_EXECUTABLE=$(command -v python3) \
    -DCMAKE_INSTALL_PREFIX=../install \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    -DDYNAMIC_LIB=ON \
    ..

  # C++ unit tests
  cmake --build . --target all-tests

  # Benchmarks
  cmake --build . --target all-benchmarks

  # Install Python library
  cmake --build . --target install

Alternatively, to build a release version with all optimizations enabled:

.. code-block:: bash

  cmake \
    -GNinja \
    -DPython_EXECUTABLE=$(command -v python3) \
    -DCMAKE_INSTALL_PREFIX=../install \
    -DCMAKE_BUILD_TYPE=Release \
    ..

  cmake --build . --target all-tests
  cmake --build . --target all-benchmarks
  cmake --build . --target install


To use the ``scippneutron`` Python module:

.. code-block:: bash

  cd ../python
  PYTHONPATH=$PYTHONPATH:../install python3

In Python:

.. code-block:: python

  import scippneutron as scn

Building using a local build of Scipp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of using a published Scipp package as part of your ``scippneutron-developer`` conda environment,
it is also possible to link ``scippneutron`` against a local build of Scipp.
To avoid conflicts, you will first need to remove the ``scipp`` entry from your generated ``scippneutron-developer.yml`` file.
Then, use the ``CMAKE_PREFIX_PATH`` to tell ``cmake`` where to find the Scipp C++ libraries:

.. code-block:: bash

  cmake \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPython_EXECUTABLE=$(command -v python3) \
    -DCMAKE_INSTALL_PREFIX=../install \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    -DDYNAMIC_LIB=ON \
    -DCMAKE_PREFIX_PATH=<your_scipp_install_dir> \
    ..

Then, simply run ``cmake --build`` as above.

Additional information
~~~~~~~~~~~~~~~~~~~~~~

For further information about additional build options, running the unit tests and building the documentation,
we refer the reader to the `developer documentation <https://scipp.github.io/reference/developer/getting-started.html>`_ of the Scipp project.


Tutorial and Test Data
~~~~~~~~~~~~~~~~~~~~~~

There are a number of data files which can be downloaded automatically by scippneutron.
The functions in `scippneutron.data` download and cache these files if and when they are used.
By default, the files are stored in the OS's cache directory.
The location can be customized by setting the environment variable ``SCIPPNEUTRON_DATA_DIR``
to the desired path.
