from typing import Dict
import numpy as np
import scipp as sc
"""
Convert a Scipp DataArray to a picklable dictionary and back.
Can be used to move DataArrays between multiprocessing.Process.
"""


def dict_dumps(data: sc.DataArray) -> Dict:
    data_dict = sc.to_dict(data)

    def _unit_and_dtype_to_str(d):
        for k, v in d.items():
            if isinstance(v, dict):
                _unit_and_dtype_to_str(v)
            else:
                if k == "unit" or k == "dtype":
                    d[k] = str(v)

    _unit_and_dtype_to_str(data_dict)
    return data_dict


def dict_loads(data_dict: Dict) -> sc.DataArray:
    def convert_from_str_unit_and_dtype(d):
        for k, v in d.items():
            if isinstance(v, dict):
                convert_from_str_unit_and_dtype(v)
            else:
                if k == "unit":
                    d[k] = sc.Unit(v)
                elif k == "dtype":
                    d[k] = np.dtype(v)

    convert_from_str_unit_and_dtype(data_dict)
    return sc.from_dict(data_dict)
