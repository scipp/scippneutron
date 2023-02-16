# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Utilities for logging in ScippNeutron.

The object returned by :func:`scippneutron.get_logger` is the only logger
used by ScippNeutron. ScippNeutron does not configure it in any way.
You are free to do so.
See also the `logging documentation <https://scipp.github.io/reference/logging.html>`_
for Scipp.
"""

import logging


def logger_name() -> str:
    """Return the name of ScippNeutron's logger."""
    return "scipp.neutron"


def get_logger() -> logging.Logger:
    """Return the logger used by ScippNeutron."""
    return logging.getLogger(logger_name())
