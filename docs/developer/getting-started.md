# Getting started

## Setting up

### Dependencies

All development environments are managed with [pixi](https://pixi.sh/).
Install it by following the [pixi installation instructions](https://pixi.sh/latest/installation/), then run

```sh
pixi install
```

from the repository root.
This creates the default environment, installs ScippNeutron into it in editable mode,
and pins everything from `pixi.lock`.
There is nothing else to install; tools such as `pandoc`, which are not on PyPI,
come from the environments as well.
See [Dependency Management](./dependency-management.md) for more information.

### Set up git hooks

The CI pipeline runs a number of code formatting and static analysis tools.
If they fail, a build is rejected.
To avoid that, you can run the same tools locally.
This can be done conveniently using [pre-commit](https://pre-commit.com/):

```sh
pixi run -e lint pre-commit install
```

Alternatively, run all of them on demand using

```sh
pixi run lint
```

Take a look at `pixi.toml` or `.pre-commit-config.yaml` to see what tools are run and how.

## Running tests

Run the tests using

```sh
pixi run test
```

Arguments are forwarded to `pytest`, so a single test file can be run with

```sh
pixi run test tests/io/cif_test.py
```

Tests that require Mantid are skipped in this environment.
They live in a separate environment because Mantid constrains the versions of
NumPy, SciPy and h5py that can be installed alongside it:

```sh
pixi run -e mantid test-mantid
```

The tests for reading and writing SQW files need a MATLAB runtime, so they too
have their own environment:

```sh
pixi run -e sqw test-sqw
```

## Building the docs

Build the documentation using

```sh
pixi run docs
```

Additionally, test the documentation using

```sh
pixi run docs doctest
pixi run docs linkcheck
```

The documentation is built in the `docs` environment, which extends the Mantid
stack because several notebooks use Mantid.

## Type checking

Static type checking is not part of CI, but can be run with

```sh
pixi run mypy
```

## Tutorial and Test Data

There are a number of data files which can be downloaded automatically by ScippNeutron.
The functions in `scippneutron.data` download and cache these files if and when they are used.
By default, the files are stored in the OS's cache directory.
The location can be customized by setting the environment variable `SCIPPNEUTRON_DATA_DIR`
to the desired path.
