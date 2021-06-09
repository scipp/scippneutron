from typing import Dict
import numpy as np
import scipp as sc
from typing import Any
"""
Convert a Scipp DataArray to a picklable dictionary and back.
Can be used to move DataArrays between multiprocessing.Process.
"""


def dict_dumps(data: sc.DataArray) -> Dict:
    data_dict = sc.to_dict(data)

    def _unit_and_dtype_to_str(d: Any):
        for k, v in d.items():
            if isinstance(v, dict):
                _unit_and_dtype_to_str(v)
            elif isinstance(v, (sc.Variable, sc.DataArray, sc.Dataset)):
                d[k] = sc.to_dict(v)
                _unit_and_dtype_to_str(d[k])
            else:
                if k == "unit" or k == "dtype":
                    d[k] = str(v)

    _unit_and_dtype_to_str(data_dict)
    return data_dict


def dict_loads(data_dict: Dict) -> sc.DataArray:
    with open("test_json.json", 'w') as jfile:
        jfile.write(str(data_dict))

    def convert_from_str_unit_and_dtype(d):
        for k, v in d.items():
            if isinstance(v, dict):
                convert_from_str_unit_and_dtype(v)
                try:
                    if str(v["dtype"]) in ("DataArray", "DataSet", "Variable"):
                        d[k] = sc.from_dict(v["value"])
                except KeyError:
                    pass
            else:
                if k == "unit":
                    d[k] = sc.Unit(v)
                elif k == "dtype":
                    try:
                        d[k] = np.dtype(v)
                    except TypeError:
                        pass  # leave DataArray etc alone

    convert_from_str_unit_and_dtype(data_dict)
    return sc.from_dict(data_dict)
