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
        delete_dtype = False
        for k, v in d.items():
            if isinstance(v, dict):
                convert_from_str_unit_and_dtype(v)
                if k not in ("attrs", "masks", "coords"):
                    d[k] = sc.from_dict(v)
            else:
                if k == "unit":
                    d[k] = sc.Unit(v)
                elif k == "dtype":
                    try:
                        d[k] = np.dtype(v)
                    except TypeError:
                        if v == "string":
                            d[k] = sc.dtype.string
                            try:
                                # Workaround for not being able to construct a
                                # variable from a scalar string numpy array.
                                # See
                                # https://github.com/scipp/scipp/issues/1974
                                d["value"] = d["value"].item()
                                delete_dtype = True
                            except AttributeError:
                                d[k] = sc.dtype.string
                        # elif any(scipp_container_type in k
                        #          for scipp_container_type in ("DataArray",
                        #                                       "DataSet",
                        #                                       "Variable")):
                        #     delete_dtype = True
        if delete_dtype:
            del d["dtype"]

    convert_from_str_unit_and_dtype(data_dict)
    return data_dict


if __name__ == "__main__":
    test_var = sc.Variable(value="commissioning", unit=sc.units.dimensionless)
    test = sc.to_dict(test_var)

    from scippneutron.file_loading.load_nexus import load_nexus
    amor_data = load_nexus("/home/matt/git/generate-nexus-files/examples"
                           "/amor/amor2020n000346_tweaked.nxs")
    amor_ser = dict_dumps(amor_data)
    out_amor = dict_loads(amor_ser)
    print(out_amor)
