.. _deployment:

Deployment
==========

The scippneutron project encompases a collection of packages in an ecosystem.
This documents the release/deployment process for each package.

Mantid Framework
----------------

Mantid is an optional runtime dependency for scippneutron and is utilized solely through its python api.
In practice it is an important component for using scippneutron for neutron scattering as it provides a number of key components, such as facility specific file loading.

For mac os and linux, ``mantid-framework`` conda packages can be produced and are placed in the anaconda ``scipp`` channel.
There is currently no windows package.

*Key Locations*

* Source code for `mantid <https://github.com/mantidproject/mantid>`_.
* Code for the `recipe <https://github.com/scipp/mantid_framework_conda_recipe>`_.
* CI `Azure pipelines <https://dev.azure.com/scipp/mantid-framework-conda-recipe/_build>`_.
* Publish location on `Anaconda Cloud <https://anaconda.org/scipp/mantid-framework>`_.

We have three azure piplelines for these packages.

Our first pipeline will build, test and publish a package to the ``scipp`` channel and is triggered by new tags in the `recipe repository <https://github.com/scipp/mantid_framework_conda_recipe>`_.
Successful pipline execution pushes new packages to `Anaconda Cloud <https://anaconda.org/scipp/mantid-framework>`_.
This is the release pipeline, and is the subject of the deployment procedure below.

Our second pipleline uses latest ``master`` of mantid to produce (but not publish) a nightly package, against our static recipie.
This allows us to anticipate and correct problems we will encounter in new package generation, and ensures we can produce new packages at short notice against an evolving mantid target, while taking into account updated depenencies on conda.

Our third pipeline is triggered by github pull requests only. This is used to produce (but no publish) packages given changes in our recipe, but against the same mantid SHA1 as is used for the tagged releases. 

Mantid Framework Deployment Procedure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Check the nightly build pipeline on azure and verify successful completion.
   If they are not, investigate and fix problems there first.
#. Determine the git revision of mantid you wish to use.
#. Compute a version string for the revision using ``tools/make_version.py`` from the recipe repository.
#. From the recipe repository update ``.azure-pipelines/pull-request.yml`` setting the ``git_rev`` and ``mantid_ver`` to match the two values from the previous steps.
#. Make a pull request with your changes and ensure that the pipeline works correctly on all platforms.
#. From the recipe repository update ``.azure-pipelines/release.yml`` setting the ``git_rev`` and ``mantid_ver`` to match the two values from the previous steps.
#. Have a peer review and merge the PR 
#. Create a new annotated tag in the recipe repository to describe the release and its purpose 
#. Push the tag to origin, which will trigger the tagged release pipeline

.. note::
  As part of the ``conda build`` step mantid's imports are tested. Packaging can therefore fail if mantid does not appear to work (import).
  
.. warning::
  When running ``conda build`` locally, ensure that ``conda-build`` is up to date (``conda update conda-build``). This can be a source of difference between what is observed on the CI (install fresh into clean conda env) and a potentially stale local environment. You should also ensure that the channel order specified is the same as is applied in the CI for your ``conda build`` command. Refer to order applied in ``conda build`` step in pipeline yaml file. Priority decreases from left to right in your command line argument order. You should also ensure that your local `~/.condarc` file does not prioritize any unexpected/conflicting channels and that flag settings such as ``channel_priority: false`` are not utilized. Note that you can set ``--override-channels`` to your ``conda build`` command to prevent local `.condarc` files getting getting in the way.
