# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Any

import matplotlib
import pytest
import scipp as sc

pytest.register_assert_rewrite('scipp.testing.assertions')


def pytest_assertrepr_compare(op: str, left: Any, right: Any) -> list[str]:
    if isinstance(left, sc.Unit) and isinstance(right, sc.Unit):
        return [f'Unit({left}) {op} Unit({right})']
    if isinstance(left, sc.DType) or isinstance(right, sc.DType):
        return [f'{left!r} {op} {right!r}']


@pytest.fixture()
def _use_ipympl():
    """
    Use ipympl interactive backend for matplotlib.
    Close figures when done, and reset matplotlib defaults.
    """
    matplotlib.use('module://ipympl.backend_nbagg')
    yield
    matplotlib.rcdefaults()
    matplotlib.use('Agg')
