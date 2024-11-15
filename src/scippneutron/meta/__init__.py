"""Metadata utilities.

.. rubric:: Models

.. autosummary::
  :toctree: ../classes
  :template: model-template.rst

  Beamline
  Person
  Software

.. rubric:: Auxiliary Classes

.. autosummary::
  :toctree: ../classes
  :template: class-template.rst

  ORCIDiD
"""

from ._model import Beamline, Person, Software
from ._orcid import ORCIDiD

__all__ = ['Beamline', 'Person', 'ORCIDiD', 'Software']
