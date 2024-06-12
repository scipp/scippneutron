# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import importlib.resources
from functools import lru_cache
from string import Template

try:
    from scipp.visualization.resources import _preprocess_style, load_dg_style
except ImportError:
    from scipp.html.resources import _preprocess_style, load_dg_style


def _read_text(filename: str, group: str) -> str:
    if hasattr(importlib.resources, 'files'):
        # Use new API added in Python 3.9
        return (
            importlib.resources.files(f'scippneutron._html_repr.{group}')
            .joinpath(filename)
            .read_text()
        )
    # Old API, deprecated as of Python 3.11
    return importlib.resources.read_text(f'scippneutron._html_repr.{group}', filename)


@lru_cache(maxsize=1)
def disk_chopper_repr_template() -> Template:
    return Template(_read_text('disk_chopper_repr.html.template', 'templates'))


@lru_cache(maxsize=1)
def disk_chopper_style() -> str:
    style = _preprocess_style(_read_text('disk_chopper.css', 'styles'))
    return load_dg_style() + f'<style>{style}</style>'
