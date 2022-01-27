# Our Interpretation of the NeXus Format

Last update: January 2022

## Introduction

This document represents our current interpretation of the [NeXus Format](https://www.nexusformat.org/) (NF).
The need for an interpretation is an unfortunate truth.
The reason for this is partially history, as the NF has evolved over time, whereas files from older versions of the NF are still used.
The second reason is unclear and/or incomplete formulations in the NF description that leave things open to interpretation or require reading between the lines.

A majority of this document is based on personal communication with Tobias Richter (TR) from the NeXus committee.
Nevertheless, most of this still represents our own interpretation.

## `NXroot`

The `NX_class` attribute for the top-level of a NeXus hierarchy (NH, often a file) should be `NXroot`.
However, this has been forgotten in the reference implementation and thus most file-writers do not include it.
It is therefore to be considered implicit.

## Relation between `NXdata` and `NXevent_data` with "subclasses"

The NF does not formally specify subclassing but, according to TR and vague hints in the manual, several classes are "similar" to `NXdata`.
This probably includes `NXdetector`, `NXlog`, and `NXmonitor` but there are likely more.

By extension, `NXevent_data` may take the role of `NXdata` for event-mode monitors or detectors.

It is unclear whether the NF requires fields and attributes of the base class (`NXdata` or `NXevent_data`) as part of the subclass (such as `NXdetector`) or whether `NXdata` or `NXevent_data` should be stored as a child.
We have observed both approach in practice.
Therefore, we go with the following interpretation:

1. *Both* approaches are permitted.
2. For a concrete group in a NH there must not be more than one `NXdata` or `NXevent_data` child (or field thereof) within the group.
   If there is more than one, the group is considered invalid.
3. Certain classes pre-define aspects that also `NXdata` could deliver.
   For example, `NXdata` uses `signal` attributes to define the field providing the signal array.
   For `NXdetector` the NF pre-defines that its signal is `data`, so if that is found it will be used, even if not `signal` attribute is present.

More concretely this means that, e.g., for loading an `NXdetector` from a NH, the implementation will:

1. Find all `NXdata` children.
2. Find all `NXevent_data` children.
3. Search the group for fields defined in `NXdata`.
   This includes looking for the `signal` attributes on the group and on fields.
4. Search the group for fields defined in `NXevent_data`.
5. Search the group for fields pre-defined in the class that are equivalents of what is defined in `NXdata`, even if the `NXdata` requirements (such as `signal` attributes) are not met.

If the above results in not more than one item, the group can be loaded.

## Bin edges

For `NXdetector` the NF defines `time_of_flight`, exceeding the data shape by one, i.e., it is meant as bin-edges.
`NXdata` does not appear to allow this explicitly.
Since what is recorded in `NXdetector` may not actually be time-of-flight, in practice this coordinate may be named differently, e.g., `time_offset`.
Therefore, we assume that this is valid in general, i.e., also for other axis tick labels (coordinates) that may be defined using the `axes` attribute.

## Missing axis labels

`NXdata` uses the `axes` attribute to define the names of fields that store coordinates for axes.
There is a legacy mechanism where the signal field has an `axes` attribute and this should not be used according to the NF.

The `axes` attribute uses `'.'` to define an axis without coordinate field.
The implication of the above is that there is no way to define the *name* of an axis.
For example, an `NXdata` group storing a stack of images may not define a coordinate for the "image" dimension of the signal dataset.
The NF does not specify any other way to define such an axis name.
We therefore have to fall back to a generic and meaningless dimension label.

Note that the `axes` attribute of the signal field could in principle be used to define such names and the `axes` attribute of the group would then only define the field names of coordinates.
That is, both attributes would work in conjunction.
It is unclear whether this is allowed or actually considered invalid in the NF.
