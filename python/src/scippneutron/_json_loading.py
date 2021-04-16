import json
import scipp as sc
from typing import Dict
import numpy as np

_filewriter_to_numpy_dtype = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64
}


def _get_units_from_attribute(obj: Dict) -> sc.Unit:
    try:
        for attribute in obj["attributes"]:
            if attribute["name"] == "units":
                return sc.Unit(attribute["values"])
    except KeyError:
        pass
    return sc.units.dimensionless


def _array_to_variable(obj: Dict):
    units = _get_units_from_attribute(obj)
    if isinstance(obj["values"], list):
        obj["values"] = sc.Variable(
            dims=[obj["name"]],
            values=np.array(obj["values"]),
            dtype=_filewriter_to_numpy_dtype[obj["dataset"]["type"]],
            unit=units)


def _attribute_arrays_to_variable(obj: Dict):
    try:
        attributes = obj["attributes"]
        for attribute in attributes:
            if isinstance(attribute["values"], list):
                attribute["values"] = sc.Variable(
                    dims=[attribute["name"]],
                    values=np.array(attribute["values"]),
                    dtype=_filewriter_to_numpy_dtype[attribute["type"]])
    except KeyError:
        pass


def object_hook(obj: Dict):
    try:
        object_type = obj["type"]
        if object_type == "dataset":
            _array_to_variable(obj)
        if object_type == "group" or object_type == "dataset":
            _attribute_arrays_to_variable(obj)
    except KeyError:
        pass
    return obj


class DecodeToScipp(json.JSONDecoder):
    """
    Deserialises JSON object
    Value arrays are deserialised to Scipp Variables
    """
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self,
                                  object_hook=object_hook,
                                  *args,
                                  **kwargs)
