from ..mantid_data_helper import MantidDataHelper
import scipp as sc
import scippneutron.mantid as mantid
import time


class MantidScippComparison:
    def __init__(self, test_description=None):
        self._test_description = test_description

    def _execute_with_timing(self, op, input):
        start = time.time()
        result = op(input)
        stop = time.time()
        return result, (stop - start) * sc.Unit('s')

    def _assert(self, a, b, allow_failure):
        try:
            assert sc.is_equal(a, b)
        except AssertionError as ae:
            if allow_failure:
                print(ae)
            else:
                raise (ae)

    def _run_from_workspace(self, in_ws, allow_failure):
        out_mantid, time_mantid = self._execute_with_timing(self._run_mantid,
                                                            input=in_ws)
        in_da = mantid.from_mantid(in_ws).astype(
            sc.dtype.float64)  # Converters set weights float32
        out_scipp, time_scipp = self._execute_with_timing(self._run_scipp,
                                                          input=in_da)

        self._assert(out_scipp, out_mantid, allow_failure)

        if isinstance(out_mantid, sc.DataArray):
            result = sc.DataArray(sc.equal(out_mantid.data, out_scipp.data),
                                  coords={
                                      'diff':
                                      out_mantid.data - out_scipp.data,
                                      'is_approx':
                                      sc.is_approx(
                                          out_mantid.data, out_scipp.data,
                                          1e-9 * sc.Unit('counts') +
                                          1e-9 * sc.abs(out_mantid.data)),
                                      'duration_scipp':
                                      time_scipp,
                                      'duration_mantid':
                                      time_mantid
                                  })
        else:
            result = out_mantid - out_scipp
        return result

    def _add_result_to_dataset(self, name, result, results):
        results[f'with_{name}' if self._test_description is None else
                f'{self._test_description}_with_{name}'] = result

    def run(self, allow_failure=False):
        import mantid.simpleapi as sapi
        results = sc.Dataset()
        for name, (hash, algorithm) in self._filenames.items():
            file = MantidDataHelper.find_file(hash, algorithm)
            print('Loading', name)
            in_ws = sapi.Load(Filename=file, StoreInADS=False)
            result = self._run_from_workspace(in_ws, allow_failure)
            self._add_result_to_dataset(name, result, results)
        for name, in_ws in self._workspaces.items():
            result = self._run_from_workspace(in_ws, allow_failure)
            self._add_result_to_dataset(name, result, results)

        return results

    @property
    def _filenames(self):
        return {}

    @property
    def _workspaces(self):
        return {}

    def _run_mantid(self, input):
        raise RuntimeError("_run_mantid not implemented in base")

    def _run_scipp(self, input):
        raise RuntimeError("_run_scipp not implemented in base")
