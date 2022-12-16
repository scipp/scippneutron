.. _installation:

Installation
============

Scippneutron requires Python 3.8 or above.

Conda
-----

The easiest way to install ``scipp`` and ``scippneutron`` is using `conda <https://conda.io>`_.
Packages from `Anaconda Cloud <https://conda.anaconda.org/scipp>`_ are available for Linux, macOS, and Windows.
It is recommended to create an environment rather than installing individual packages.

With the provided environment file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Download :download:`scippneutron.yml <../environments/scippneutron.yml>` for the stable release version of scipp.
2. In a terminal run:

   .. code-block:: sh

      conda activate
      conda env create -f scippneutron.yml
      conda activate scippneutron
      jupyter lab

   The ``conda activate`` ensures that you are in your ``base`` environment.
   This will take a few minutes.
   Above, replace ``scippneutron.yml`` with the path to the download location you used to download the environment.
   Open the link printed by Jupyter in a browser if it does not open automatically.

If you have previously installed scipp with conda we nevertheless recommend creating a fresh environment rather than trying to ``conda update``.
You may want to remove your old environment first, e.g.,

.. code-block:: sh

   conda activate
   conda env remove -n scippneutron

and then proceed as per instructions above.
The ``conda activate`` ensures that you are in your ``base`` environment.

Without the provided environment file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a new conda environment with scippneutron:

.. code-block:: sh

   $ conda create -n env_with_scipp -c conda-forge -c scipp -c ess-dmsc scippneutron

.. note::
   Instaling ``scippneutron`` on Windows requires ``Microsoft Visual Studio 2019 C++ Runtime`` (and versions above) installed.
   Visit https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads for the up to date version of the library.

After installation the modules ``scipp`` and ``scippneutron`` can be imported in Python.
Note that only the bare essential dependencies are installed.
If you wish to use plotting functionality you will also need to install ``matplotlib`` and ``ipywidgets``.
If you wish to use the live data functionality you will need to install ``conda-forge::python-confluent-kafka`` and ``ess-dmsc::ess-streaming-data-types`` with conda, or on Windows install ``confluent-kafka`` and ``ess-streaming-data-types`` from the PyPI:

.. code-block:: sh

   $ pip install confluent-kafka ess-streaming-data-types

To update or remove ``scippneutron`` use `conda update <https://docs.conda.io/projects/conda/en/latest/commands/update.html>`_ and `conda remove <https://docs.conda.io/projects/conda/en/latest/commands/remove.html>`_.

If you wish to use ``scippneutron`` with Mantid you may use the following command to create an environment containing both ``scippneutron`` and Mantid framework.

Note that Conda packages for Mantid are only available on Linux and macOS, and are currently maintained separate to the Mantid project.
This is due to some dependencies being too old to work in the same environment as Scipp.

.. code-block:: sh

  $ conda create \
      -n env_with_scipp_and_mantid \
      -c conda-forge \
      -c scipp \
      python=3.8 \
      scippneutron \
      mantid-framework

.. note::
   Instaling `scippneutron`` with Mantid on Windows is possible but requires ``Windows Subsystem for Linux 1`` (WSL 1) installed and is limited to Windows 10.
   Please follow the steps on the `Windows Subsystem for Linux Installation Guide page <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_
   to enable Linux support.
   Once ``WSL 1`` is installed, setting up Scipp with Mantid follows the Linux specific directions described above.

Pip
---

ScippNeutron is available from `PyPI <https://pypi.org/>`_ via ``pip``:

.. code-block:: sh

   pip install scippneutron

Note that Mantid is not available via ``pip``.
If you depend on it, you will need to install it separately.
We recommending installation of ScippNeutron via ``conda`` in this case (see above).
