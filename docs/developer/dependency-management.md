# Dependency management

scippneutron is a library, so the package dependencies are never pinned.
Lower bounds are fine and individual versions can be excluded.
See, e.g., [Should You Use Upper Bound Version Constraints](https://iscinumpy.dev/post/bound-version-constraints/) for an explanation.
Those dependencies live in `pyproject.toml` and are the only ones users are exposed to.

Development dependencies (as opposed to dependencies of the deployed package that users need to install) are pinned to an exact version in order to ensure reproducibility.
This also includes dependencies used for the various CI builds.
This is done with [pixi](https://pixi.sh/): `pixi.toml` declares the environments and `pixi.lock` pins every package in them for all supported platforms.
Both files are committed.

## Why pixi

Mantid is only distributed as a conda package, while the rest of the stack is on PyPI, so development requires resolving across both ecosystems at once.
Pixi does this in a single solve and writes the result to one lock file.
That is why it replaced the previous combination of `pip-compile-multi`, `tox` and a hand-maintained conda environment file, which had to keep two sets of pins in agreement by hand.
The Mantid feature in `pixi.toml` only names packages that must come from conda;
their supported version ranges remain owned by `pyproject.toml` and are checked by
Pixi's combined conda/PyPI solve.

## Environments

| Environment | Purpose |
|-------------|---------|
| `default`   | Tests, resolved from PyPI, i.e. the way a user installing with pip gets it |
| `mantid`    | Mantid tests, resolved from conda so the stack is ABI-consistent with Mantid |
| `docs`      | Building the documentation; extends `mantid` because several notebooks use it |
| `sqw`       | SQW/Horace tests, which need `pace_neutrons` and a MATLAB runtime |
| `lint`      | Formatting and static analysis |
| `build`     | Building the sdist and wheel |

List them, and the tasks they provide, with `pixi task list`.

Note that Mantid is not available for Windows in combination with scipp; see the comment in `pixi.toml` for details.

## Updating dependencies

To re-resolve everything, run

```sh
pixi update
```

and commit the resulting `pixi.lock`.
To update a single package, name it, e.g. `pixi update scipp`.

The workspace sets `exclude-newer` to two weeks, so a resolution ignores packages published more recently than that.
This keeps a freshly published, occasionally broken build from breaking the lock the moment it lands.
The Scipp stack is exempted from that delay via the `[exclude-newer]` and `[pypi-exclude-newer]` tables, because those are the packages we want to test against as soon as they are released.
The exemptions only have an effect because of the workspace-level baseline.

## Testing outside the lock file

Pinned environments cannot catch a dependency range that is wrong.
The nightly workflow therefore also resolves dependencies from scratch with [uv](https://docs.astral.sh/uv/):

- `--resolution=lowest-direct` checks that the lower bounds in `pyproject.toml` are actually correct.
- `--resolution=highest` checks against the newest release of everything, both on the main branch and at the last release tag.
- A separate upstream solve installs the Scipp nightly wheel and the Git `main`
  branches of the rest of the Scipp stack. It includes the optional and SQW
  dependencies and runs with MATLAB so the Horace integration cannot silently
  skip.
