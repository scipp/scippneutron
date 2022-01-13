# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

from contextlib import contextmanager
import logging
import re
from typing import Callable, Optional, Union

import pytest


class LogCheck:
    _Filter = Callable[[logging.LogRecord], bool]

    def __init__(self, _caplog):
        self._caplog = _caplog

    @staticmethod
    def _build_regex_filter(x: Union[str, re.Pattern], attr: str):
        pattern = re.compile(x) if not isinstance(x, re.Pattern) else x
        return lambda record: pattern.search(getattr(record, attr)) is not None

    @staticmethod
    def _build_level_filter(level: Union[int, str, re.Pattern]) -> _Filter:
        if isinstance(level, int):
            return lambda record: record.levelno == level
        return LogCheck._build_regex_filter(level, 'levelname')

    @staticmethod
    def _build_logger_filter(logger: Union[str, logging.Logger, re.Pattern]) -> _Filter:
        logger = logger.name if isinstance(logger, logging.Logger) else logger
        return LogCheck._build_regex_filter(logger, 'name')

    @staticmethod
    def _build_message_filter(message: Union[str, re.Pattern]) -> _Filter:
        return LogCheck._build_regex_filter(message, 'message')

    def _fail(self, level, logger, message):
        filter_msg = []
        if level is not None:
            filter_msg.append(f"level: '{level}'")
        if logger is not None:
            filter_msg.append(f"logger: '{logger}'")
        if message is not None:
            filter_msg.append(f"message: '{message}'")
        captured_msg = ",\n".join(map(repr, self._caplog.records))
        pytest.fail(f'DID NOT LOG. No logs matching [{", ".join(filter_msg)}].\n'
                    f'Recorded logs: [{captured_msg}]')

    @contextmanager
    def logs(self,
             *,
             level: Union[int, str, re.Pattern],
             logger: Optional[Union[str, logging.Logger, re.Pattern]] = None,
             message: Optional[Union[str, re.Pattern]] = None):
        try:
            with self._caplog.at_level(level=level, logger=logger):
                yield
        finally:
            filters = []
            if level is not None:
                filters.append(self._build_level_filter(level))
            if logger is not None:
                filters.append(self._build_logger_filter(logger))
            if message is not None:
                filters.append(self._build_message_filter(message))

            records = [
                record for record in self._caplog.records if all(
                    filt(record) for filt in filters)
            ]

            if not records:
                self._fail(level, logger, message)


@pytest.fixture
def logcheck(caplog):
    """
    Test fixture to check if code logs a specific message.
    """
    return LogCheck(caplog)
