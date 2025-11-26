# SQW

This document describes the **SQW format v4.0**.

SQW files are used in spectroscopy by [Horace](https://pace-neutrons.github.io/Horace) for reduced data and intermediate analysis results.
ScippNeutron can write and, to a limited degree, read SQW files using {mod}`scippneutron.io.sqw`.

```{note}
There is no official specification for the format of SQW files.
The contents of this document are based on what documentation is available, on reading the source code of Horace, and on experiments with Horace and SQW files.
This document should *not* be regarded as authoritative.
```

See, e.g., [07-sqw_redesign.md](https://github.com/pace-neutrons/Horace/blob/3c00958d578c625a7f2274ef7b34557578abb934/documentation/add/07-sqw_redesign.md) for some documentation provided by the developers of Horace.

## Overview

SQW files are binary files that encode neutron counts both in a flexible binned format called 'pixels' and a multidimensional histogram called 'DND' (or D1D through D4D for concrete cases).
In addition to the data, SQW files store metadata that describes how the experiment was set up and how the data was binned.

### Data Blocks

Data and metadata are organized into 'blocks' that are stored separately in the file.
In principle, SQW files can store any number of blocks with arbitrary content.
But in practice, Horace requires the following blocks and allows only the following blocks.
Block names are split into a 'name' and a 'level2_name' which in ScippNeutron are combined into a tuple.

:::{list-table}
:header-rows: 1

* - Name
  - Level 2 name
  - Content
* -
  - main_header
  - General information about the file. ({class}`SqwMainHeader <scippneutron.io.sqw.SqwMainHeader>`)
* -
  - detpar
  - Used for data analysis, in particular [Tobyfit](https://pace-neutrons.github.io/Horace/v4.0.0/manual/Tobyfit.html). ScippNeutron leaves this block empty.
* - data
  - metadata
  - Information about how the nd data was histogrammed and how to display it. ({class}`SqwDndMetadata <scippneutron.io.sqw.SqwDndMetadata>`)
* - data
  - nd_data
  - Histogram data.
* - experiment_info
  - instruments
  - Information about the instrument where the data was measured. One {class}`SqwIXNullInstrument <scippneutron.io.sqw.SqwIXNullInstrument>` per [run](#Runs). (Horace defines several instrument classes. But so far, the null instrument has been sufficient.)
* - experiment_info
  - samples
  - Information on the sample that was measured. One {class}`SqwIXSample <scippneutron.io.sqw.SqwIXSample>` per [run](#Runs).
* - experiment_info
  - expdata
  - Experimental settings such as sample rotation. One {class}`SqwIXExperiment <scippneutron.io.sqw.SqwIXExperiment>` per [run](#Runs).
* - pix
  - metadata
  - Information about the pixel data. Largely hidden and automated in ScippNeutron.
* - pix
  - data_wrap
  - 'Pixel' data, i.e., binned observations as a function of momentum and energy transfer. Stored as a $9 \times N$ array with these [rows](https://github.com/pace-neutrons/Horace/blob/master/documentation/add/05_file_formats.md) in order:
    - `u1`, `u2`, `u3`, `u4`: usually momentum transfers $Q_x$, $Q_y$, $Q_z$ and energy transfer $\Delta E$.
    - `irun`: [Run](#Runs) index.
    - `idet`: Detector (pixel) index.
    - `ien`: Energy bin index.
    - `signal`: Counts.
    - `error`: Variances of `signal`.
:::

### Runs

Data can be collected in multiple runs or 'settings' where each run has a fixed set of sample and instrument parameters.
In SQW, runs are encoded by assigning a run index `irun` to each observation and providing an appropriate instrument, sample, and experiment object.
The instrument and sample objects are often the same for all runs.
This can be encoded by way of 'unique object containers' which store only one copy of each unique object and indices to associate these objects with runs.

### FORTRAN Layout

Horace, being a MATLAB program, uses FORTRAN order for arrays and 1-based indexing.
{mod}`scippneutron.io.sqw` converts between these conventions and the Python conventions automatically in most cases.
But some data, most notably the pixel array, must be provided with 1-based indices.

### Byte Order

SQW files do not have a dedicated mechanism for encoding or detecting byte order.
Horace seems to ignore the issue and seems to always read and write files in the host machine's native order.
ScippNeutron supports saving files with native, little, or big endianness.
It also attempts to detect the byte order of a file by inspecting the [header](#File-Header).

## File Structure

SQW files contain the following in order:
1. [File header](#file-header)
2. [Block allocation table](#block-allocation-table)
3. [Blocks](#block-storage)

### File Header

SQW files begin with a file header.
For Horace version 4.0, this header is
```text
| program name | program version | file type | n dims  |
| :char array  | :float64        | :uint32   | :uint32 |
```
- The program name is a [character array](#character-array) with value `"horace"`.
  (Note that the first 4 bytes in the file are the array length.)
- The program version is a [float64](#number-uintx-floatx) with value `4.0`.
- The file type can be either
  - `0` for DND files or
  - `1` for SQW files. (Required by ScippNeutron.)
- The number of dimensions indicates how many rows `u1` - `u4` in the pixel data are used.

### Block Allocation Table

The block allocation table (BAT) encodes which [data blocks](#data-blocks) exist and where they are located in the file.
The BAT is a sequence of data block descriptors.
The structure of the BAT is
```text
| size | n_blocks | descriptor_0 | ... | descriptor_n
| :u32 | :u32     | :descriptor  | ... | :descript
```
where the `:descriptor` type is
```text
| block_type  | name        | level2_name | position | size | locked |
| :char array | :char array | :char array | :u64     | u32  | u32    |
```
- `Bat.size`: Size in bytes of the BAT.
- `Bat.n_blocks`: The number of blocks.
- The descriptors contain
  - `descriptor.block_type`: One of
    - `"data_block"`
    - `"dnd_data_block` (For `("data", "nd_data")`)
    - `"pix_data_block"` (For `("pix", "data_wrap")`)
  - `descriptor.name` and `descriptor.level2_name`: The block's name.
  - `descriptor.position`: Offset in bytes from the start of the file where the block is located.
  - `descriptor.size`: Size in bytes of the block.
  - `descriptor.locked`: Interpreted as a boolean. If true, the block is currently locked for writing and should not be accessed.

### Block Storage

Blocks are stored after the BAT in any order.
How blocks are stored depends on their type.

#### Regular Data Blocks

Blocks with type `"data_block"` are stored as a single [object array](#object-array) or [self serializing object](#self-serializing).

#### DND Data Blocks

Blocks with type `"dnd_data_block"` are stored as a
```text
| ndim | length_0 | length_1 | ... | values     | errors     | counts     |
| :u32 | :u32     | :u32     | ... | :f64 array | :f64 array | :u64 array |
```
- `ndim` and `length_i` encode a [shape](#shape) but uses a `u32` for `ndim` instead of the usual `u8`.
- `values` is the average intensity per bin (`pix.bins.mean()` or, equivalently, `pix.bins.sum() / pix.bins.size()`).
- `errors` is the standard deviation per bin (`sc.stddevs(pix.bins.mean())`).
- `counts` is the number of observations in each bin (`pix.bins.size()`).

Note the example code at the end of the next section.

#### Pixel Data Blocks

Blocks with type `"pix_data_block"` are stored as a $9 \times N$ array in column-major layout, where $N$ is the number of pixels.
See [Data Blocks](#data-blocks) for the list of rows.
The column-major layout results in a file of the form
```text
| u1[0] | u2[0] | u3[0] | u4[0] | irun[0] | idet[0] | ien[0] | signal[0] | error[0]
| u1[1] | u2[1] | u3[1] | u4[1] | irun[1] | idet[1] | ien[1] | signal[1] | error[1]
| ...
```

Pixels are sorted in bins following the same order as in the [DND data block](#dnd-data-blocks).
Given the [FORTRAN layout](#fortran-layout), the pixels are sorted according to
```python
sorted_bins = binned_observations.transpose(['u4', 'u3', 'u2', 'u1'])
```
Within each bin, the data is unordered.

The number of bins is encoded by the `n_bins_all_dims` attribute of the DND metadata block.
The number of pixels in each bin is determined by the ``counts`` array in the [DND data block](#dnd-data-blocks).
And the ranges of the ``u``s are determined by the ``img_range`` attribute of the DND metadata block.
```python
import itertools
import numpy as np
import scipp as sc
from scippneutron.io import sqw

with sqw.Sqw.open("file.sqw") as f:
    metadata = f.read_data_block("data", "metadata")
    dnd = f.read_data_block("data", "nd_data")
    pix = f.read_data_block("pix", "data_wrap")

n_pix = dnd[2].astype(int)

# Compute the bin edges:
img_range = metadata.axes.img_range
n_bins = metadata.axes.n_bins_all_dims.values
u_edges = [
    sc.linspace(f'u{i}', img_range[i][0], img_range[i][1], n_bins[i] + 1)
    for i in range(len(n_bins))
]

# Access pixels bin-by-bin:
bin_offsets = np.r_[0, np.cumsum(n_pix.flat)]
for bin_index, (l, k, j, i) in enumerate(  # noqa: E741
    itertools.product(*(range(nb) for nb in n_bins[::-1]))
):
    pix_slice = slice(bin_offsets[bin_index], bin_offsets[bin_index + 1])
    observations_in_bin = pix[pix_slice]
```

## Data Types

This section describes the data types used in SQW files.

### Logical

A boolean.
Encoded as a single byte that is `\x00` for false and anything else (typically `\0x01`) for true.

### Number (uintx, floatx)

All numbers are encoded as their byte representation in the target byte order.

### Character Array

A sequence of ASCII characters including the length.
Encoded as characters ``c0`` through ``cn``:
```text
| length  | c0     | c1     | ... |
| :uint32 | :uint8 | :uint8 | ... |
```

### Fixed Character Array

Like [Character Array](#character-array) but the length is not stored in the file.
So the length must be known from context.

### Struct

A collection of named fields with different types.

The field names are encoded as [Fixed Character Arrays](#fixed-character-array) with the lengths up front.
The field values are encoded as a [Cell Arrays](#cell-array).
```text
| n_fields | name_len_0 | ... | name_len_n |
| :uint32  | :uint32    | ... | :uint32    |

| name_0            | ... | name_n            |
| :fixed_char_array | ... | :fixed_char_array |

| value_0     | ... | value_n     |
| :cell_array | ... | :cell_array |
```

When used in an [Object Array](#object-array) with more than one element, the fields are stored in a 'struct of arrays' fashion.
This means that if the surrounding object array has shape `S`, so does each [cell array](#cell-array).
For example, given structs
```python
class MyStruct:
    i: int
    decimal: float

data = [MyStruct(i=10, decimal=0.1), MyStruct(i=20, decimal=0.2)]
```
the data will be encoded as
```text
| 2 | 1 | 7 | "i" | "decimal" | 10 | 20 | 0.1 | 0.2 |
```

### Cell Array

Cell arrays are MATLAB’s way of encoding heterogeneous arrays where the elements can have different types.
Each element is encoded as an [object array](#object-array).

In SQW, cell arrays can occur either standalone or within an object array (by way of a struct).
Standalone cell arrays encode their shape in the same way as an object array.
Otherwise, they do *not* store their shape; it needs to be provided by the [object array](#object-array) that the cell array is part of.

### Object Array

Object arrays are homogeneous, multidimensional arrays of any object type.
They encode their element type using a type tag:
```text
| type_tag | shape  | data |
| :uint8   | :Shape | :Any |
```
See ``TypeTag`` in the source code for a list of supported types and their tags.

Object arrays have some non-trivial interactions with different types.

- For **simple types** like numbers, logical, and strings, ``data`` is simply an array of the given type in FORTRAN layout.
- For **structs**, the data is encoded in a 'struct of arrays' fashion as described in the [Struct](#struct) section.
- For **cell arrays**, i.e., if the type tag indicated 'cell array', the entire object array is instead processed as a [cell array](#cell-array). This means that ``data`` is an array of object arrays.
- For **self-serializing** types, there is no shape and ``data`` has to encode how to read it.

### Self Serializing

Some data is 'self-serializing', meaning that it does not rely on an external type tag and shape to describe the data.
Instead, if used inside an object array, this data must contain its own type tag.
In ScippNeutron, this case is treated like a nested object array.

### Auxiliary Types

These types are not represented by type tags but occur in fixed locations in the file.

#### Shape

The shape of a multidimensional array.
Encoded as
```text
| ndim | length_0 | length_1 | ... |
| :u8  | :u32     | :u32     | ... |
```
where `ndim` is the number of dimensions and there are this many `length_i` entries.

#### Data Block Type

Each data block is encoded as a struct with, among others, fields 'serial_name' and 'version' which together identify the block.
This is used to determine how to deserialize the block into a Python model.

#### Unique Object Container

'Unique object container' and 'Unique reference container' are always used together and represent an array of objects where duplicates are only stored once.
Each 'Unique reference container' contains a 'Unique object container' which in turn contains an array of indices that map from array index to a unique object.
