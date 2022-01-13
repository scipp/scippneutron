# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
import logging


def get_logger() -> logging.Logger:
    """Return scippneutron's logger.

    The logger is a child of scipp's logger.
    """
    return logging.getLogger('scipp.neutron')
