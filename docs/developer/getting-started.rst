Getting Started
===============

What goes into scippneutron
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Code contributions to scippneutron should meet the following crieria:

* Relate to neutron scattering data reduction in general or be particular to a technique
* Will *not* contain any instrument or facility specific assumptions or branching
* Will provide unit tests, which demostrate a fixed regression or exercise a new feature
  - Unit test suites will execute rapidly, no slow tests
  - As far as possible, unit test will make no use of external data via IO

In addition, all code should meet the code :ref:`conventions`.
