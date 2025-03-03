# API Reference

## Top-level functions

### Mantid Compatibility

```{eval-rst}
.. currentmodule:: scippneutron

.. autosummary::
   :toctree: ../generated/functions
   :recursive:

   from_mantid
   load_with_mantid
   to_mantid
   fit
```

### Coordinate transformations (Unit conversion)

```{eval-rst}
.. autosummary::
   :toctree: ../generated/functions
   :recursive:

   convert
```

### Beamline geometry

Note that `theta` or `scattering_angle` are deliberately not supported,
due to some ambiguity on how the terms are used in the community
and possible confusion of `theta` (from Bragg’s law) with `theta` in spherical coordinates.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/functions
   :recursive:

   position
   source_position
   sample_position
   Ltotal
   L1
   L2
   two_theta
   incident_beam
   scattered_beam
```

## Submodules

```{eval-rst}
.. autosummary::
   :toctree: ../generated/modules
   :template: module-template.rst
   :recursive:

   atoms
   absorption
   chopper
   conversion
   io
   logging
   metadata
   peaks
   tof
```
