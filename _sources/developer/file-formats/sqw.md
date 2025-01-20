# SQW

This document describes the **SQW format v4.0**.

SQW files are used in spectroscopy by [Horace](https://pace-neutrons.github.io/Horace) for reduced data and intermediate analysis results.
ScippNeutron can write and, to a limited degree, read SQW files using {mod}`scippneutron.io.sqw`.

```{note}
There is no official specification for the format of SQW files.
The contents of this document are based the little bits of documentation that are available, from reading the source code of Horace, and from experimenting with Horace and SQW files.
It should *not* be regarded as authoritative.
```

## Overview

SQW files are binary files that encode neutron counts both in a flexible binned format called 'pixels' and a multidimensional histogram called 'DND' (or D1D through D4D for concrete cases).
In addition to the data, SQW files store metadata that describe how the experiment was set up and how the data were binned.

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
  - Information about the instrument where the data were measured. One {class}`SqwIXNullInstrument <scippneutron.io.sqw.SqwIXNullInstrument>` per [run](#Runs). (Horace defines several instrument classes but so far, we have only use the null instrument.)
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
  - 'Pixel' data, i.e., binned observations as a function of momentum and energy transfer. Stored as a $9 \times N$ array with [rows](https://github.com/pace-neutrons/Horace/blob/master/documentation/add/05_file_formats.md):
    - `u1`, `u2`, `u3`, `u4`: usually momentum transfers $Q_x$, $Q_y$, $Q_z$ and energy transfer $\Delta E$.
    - `irun`: [Run](#Runs) index.
    - `idet`: Detector (pixel) index.
    - `ien`: Energy bin index.
    - `signal`: Counts.
    - `error`: Variances of `signal`.
:::

### Runs

Data can be collected in multiple runs or 'settings' where each run has a fixed set of sample parameters.
In SQW, runs are encoded by assigning a run index `irun` to each observation and providing an appropriate instrument, sample, and experiment object.
The instrument and sample objects are often the same for all runs.
This can be encoded by way of 'unique object containers' which store only one copy of each unique object and indices to associate these objects with runs.

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

The block allocation table (BAT) encodes which data blocks exist and where they are located in the file.
The BAT is a sequence of data block descriptors.
The structure of the BAT is
```text
| size | n_blocks | descriptor  | ...
| :u32 | :u32     | :descriptor | ...
```
where `descriptor` is
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
- `values` currently unknown but looks like a spectroscopy pattern.
- `errors` currently unknown, uncertainties of `values`?
- `counts` currently unknown but looks like a spectroscopy pattern.

#### Pixel Data Blocks

Blocks with type `"pix_data_block"` are stored as a $9 \times N$ array in column-major layout, where $N$ is the number of pixels.
See [Data Blocks](#data-blocks) for the list of rows.
The column-major layout results in a file of the form
```text
| u1[0] | u2[0] | u3[0] | u4[0] | irun[0] | idet[0] | ien[0] | signal[0] | error[0]
| u1[1] | u2[1] | u3[1] | u4[1] | irun[1] | idet[1] | ien[1] | signal[1] | error[1]
| ...
```

Columns must be sorted in ascending order according to the `u` rows where `u1` is sorted, for each constant `u1`, `u2` is sorted, etc.

## Objects

### Logical

A boolean.
Encoded as a single byte that is `\x00` for false and anything else (typically `\0x01`) for true.

### Number (uintx, floatx)

All numbers are encoded as their byte representation in the target byte order.

### Character Array

A sequence of ASCII characters with *known* length.
Encoded as
```text
| length  | c0     | c1     | ... |
| :uint32 | :uint8 | :uint8 | ... |
```

### Fixed Character Array

A sequence of ASCII characters with *unknown* length.
Like [Character Array](#character-array) but the length is not stored in the file.

### Shape

The shape of a multidimensional array.
Encoded as
```text
| ndim | length_0 | length_1 | ... |
| :u8  | :u32     | :u32     | ... |
```
where `ndim` is the number of dimensions and there are this many `length_i` entries.

### Struct

TODO

### Object Array

### Self Serializing
