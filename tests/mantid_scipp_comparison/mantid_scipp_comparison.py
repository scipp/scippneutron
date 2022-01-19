import scipp as sc
import scippneutron.mantid as mantid
from scippneutron.data import get_path
import time
from abc import ABC, abstractmethod


class MantidScippComparison(ABC):
    def __init__(self, test_description=None):
        self._test_description = test_description

    @staticmethod
    def _execute_with_timing(op, input):
        start = time.time()
        result = op(input)
        stop = time.time()
        return result, (stop - start) * sc.Unit('s')

    @staticmethod
    def _assert(a, b):
        rtol = 1e-9 * sc.units.one
        atol = 1e-9 * a.unit
        if isinstance(a, sc.DataArray):
            assert (sc.allclose(a.data, b.data, rtol=rtol, atol=atol)
                    and sc.utils.comparison.isnear(
                        a, b, rtol=1e-6 * sc.units.one, include_data=False))
        else:
            assert sc.all(sc.isclose(a, b, rtol=rtol, atol=atol)).value

    def _run_from_workspace(self, in_ws):
        out_mantid, time_mantid = self._execute_with_timing(self._run_mantid,
                                                            input=in_ws)
        in_da = mantid.from_mantid(in_ws)
        if in_da.data.bins is not None:
            in_da = in_da.astype(sc.DType.float64)  # Converters set weights float32
        out_scipp, time_scipp = self._execute_with_timing(self._run_scipp, input=in_da)

        self._assert(out_scipp, out_mantid)

        return {'scipp': out_scipp, 'mantid': out_mantid}

    def _append_result(self, name, result, results):
        results[f'with_{name}' if self._test_description is None else
                f'{self._test_description}_with_{name}'] = result

    def run(self):
        import mantid.simpleapi as sapi
        results = {}
        if self._filenames == {} and self._workspaces == {}:
            raise RuntimeError('No _files or _workspaces provided for testing ')
        for filename in self._filenames:
            file = get_path(filename)
            print('Loading', filename)
            in_ws = sapi.Load(Filename=file, StoreInADS=False)
            result = self._run_from_workspace(in_ws)
            self._append_result(filename, result, results)
        for name, in_ws in self._workspaces.items():
            result = self._run_from_workspace(in_ws)
            self._append_result(name, result, results)

        return results

    @property
    def _filenames(self):
        return {}

    @property
    def _workspaces(self):
        return {}

    @abstractmethod
    def _run_mantid(self, in_ws):
        pass

    @abstractmethod
    def _run_scipp(self, in_da):
        pass
