.. _contributing:

Contributing to scippneutron
============================

Overview
--------

Contributions, bug reports, and ideas are always welcome.
The following section outlines the scope of scippneutron.
If in doubt whether a feature falls within the scope of scippneutron please `ask on github <https://github.com/scipp/scippneutron/issues>`_ before implementing functionality, to reduce the risk of rejected pull requests.
Asking and discussing first is generally always a good idea, since our road map is not very mature at this point.

Scope
-----

``scippneutron`` shall contain only generic neutron-specific functionality.
Facility-specific or instrument-specific functionality must not be added.
Examples of generic functionality that is permitted are 

* Unit conversions, which could be generic for all time-of-flight neutron sources.
* Published research such as absorption corrections.

Examples of functionality that shall not be added to ``scippneutron`` are handling of facility-specific file types or data layouts, or instrument-specific correction algorithms.
`ess <https://github.com/scipp/ess>`_ is an example codebase providing facility-specific algorithms
Security
--------

Given the low (yet non-zero) chance of an issue in scippneutron that affects the security of a larger system, security related issues should be raised via GitHub issues in the same way as "normal" bug reports.
