# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
from typing import Dict
import numpy as np
import scipp as sc
from typing import Any
"""
Convert a Scipp DataArray to a picklable dictionary and back.
Can be used to move DataArrays between multiprocessing.Process.
"""

_scipp_containers = ("DataArray", "DataSet", "Variable")


def convert_to_pickleable_dict(data: sc.DataArray) -> Dict:
    data_dict = sc.to_dict(data)

    def _unit_and_dtype_to_str(d: Any):
        for k, v in d.items():
            if isinstance(v, dict):
                _unit_and_dtype_to_str(v)
            elif k == "unit":
                d[k] = str(v)
            elif k == "dtype":
                d[k] = str(v)
                if any(scipp_container_type == d[k]
                       for scipp_container_type in _scipp_containers):
                    d["values"] = sc.to_dict(d["values"])
                    _unit_and_dtype_to_str(d["values"])

    _unit_and_dtype_to_str(data_dict)
    return data_dict


def convert_from_pickleable_dict(data_dict: Dict) -> sc.DataArray:

    def convert_from_str_unit_and_dtype(d):
        delete_dtype = False
        for k, v in d.items():
            if isinstance(v, dict):
                convert_from_str_unit_and_dtype(v)
                if k not in ("attrs", "masks", "coords"):
                    if {"coords", "data"}.issubset(set(v.keys())):
                        # from_dict does not work with nested DataArrays,
                        # so we have to manually construct DataArrays here.
                        d[k] = sc.DataArray(coords=v["coords"],
                                            data=v["data"],
                                            attrs=v["attrs"])
                    else:
                        try:
                            if any(scipp_container_type in str(v["dtype"])
                                   for scipp_container_type in _scipp_containers):
                                del v["dtype"]
                        except KeyError:
                            pass
                        d[k] = sc.from_dict(v)
            else:
                if k == "dtype":
                    try:
                        d[k] = np.dtype(v)
                    except TypeError:
                        if v == "string":
                            d[k] = sc.DType.string
                        elif any(scipp_container_type in k
                                 for scipp_container_type in _scipp_containers):
                            delete_dtype = True
        # Delete now, not while looping through dictionary
        if delete_dtype:
            del d["dtype"]

    convert_from_str_unit_and_dtype(data_dict)
    return sc.DataArray(data=data_dict["data"],
                        coords=data_dict["coords"],
                        attrs=data_dict["attrs"])
