# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones
import typing
from dataclasses import dataclass
from typing import Union, Dict, Tuple, Optional, TypeVar, Callable, Any, Type
import h5py


class BadSource(Exception):
    """
    Raise if something is wrong with data source which
    prevents it being used. Warn the user.
    """
    pass


class SkipSource(Exception):
    """
    Raise to abort using the data source, do not
    warn the user.
    """
    pass


class MissingDataset(Exception):
    pass


class MissingAttribute(Exception):
    pass


@dataclass
class Group:
    """
    This class exists because h5py.Group has a "parent" property,
    but we also need to access the parent when parsing Dict
    loaded from json
    """
    group: Union[h5py.Group, Dict]
    parent: Union[h5py.Group, Dict]
    path: str
    contains_stream: bool = False


T = TypeVar("T")


def try_or_default(function: Callable[..., T],
                   exceptions: Tuple[Type[Exception]],
                   default: T,
                   func_args: Optional[Tuple[Any, ...]] = None,
                   func_kwargs: Dict[str, typing.Any] = None) -> T:
    """
    Tries to call a specified function with the provided arguments.
    If the function fails with one of the provided exceptions,
    silently return the default.

    Args:
        function: The function to be called
        exceptions: A tuple of exceptions to catch
        default: The default value to return if one of the specified
            exceptions is thrown.
        func_args: Tuple of arguments to pass to the provided function
        func_kwargs: Dictionary of keyword arguments to pass to the
            provided function

    Returns:
        The result of the function if no specified exception was thrown,
        or the default value if one of the specified exceptions was thrown.

        If an exception is thrown that was not listed, this exception is
        propagated.
    """
    try:
        return function(*(func_args or ()), **(func_kwargs or {}))
    except exceptions:
        return default
