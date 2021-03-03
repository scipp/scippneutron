from ..mantid_data_helper import MantidDataHelper
import scipp as sc
import scippneutron.mantid as mantid
import time
from abc import ABC, abstractmethod


class MantidScippComparison(ABC):
    def __init__(self, test_description=None):
        self._test_description = test_description

    def _execute_with_timing(self, op, input):
        start = time.time()
        result = op(input)
        stop = time.time()
        return result, (stop - start) * sc.Unit('s')

    def _fuzzy_compare(self, a, b, tol):
        same_data = sc.all(sc.is_approx(a.data, b.data, tol)).value
        if not len(a.meta) == len(b.meta):
            raise RuntimeError('Different number of items'
                               f'in meta {len(a.meta)} {len(b.meta)}')
        for key, val in a.meta.items():
            x = a.meta[key]
            y = b.meta[key]
            if x.shape != y.shape:
                raise RuntimeError(f'For meta {key} have different'
                                   f' shapes {x.shape}, {y.shape}')
            if val.dtype in [sc.dtype.float64, sc.dtype.float32]:
                if sc.sum(~sc.isfinite(x)).value > 0 or sc.sum(
                        ~sc.isfinite(y)).value > 0:
                    raise RuntimeError(
                        f'For meta {key} have non-finite entries')
                coord_tol_factor = 1e-6
                vtol = sc.abs(
                    a.meta[key]
                ) * coord_tol_factor + coord_tol_factor * a.meta[key].unit
                if not sc.all(sc.is_approx(a.meta[key], b.meta[key],
                                           vtol)).value:
                    return False
        return same_data

    def _assert(self, a, b, allow_failure):
        try:
            dtol_factor = 1e-9
            if isinstance(a, sc.DataArray):
                tol = dtol_factor * a.data.unit + dtol_factor * sc.abs(a.data)
                assert self._fuzzy_compare(a, b, tol)
            else:
                tol = dtol_factor * a.unit + dtol_factor * sc.abs(a)
                assert sc.all(sc.is_approx(a, b, tol)).value
        except AssertionError as ae:
            if allow_failure:
                print(ae)
            else:
                raise (ae)

    def _run_from_workspace(self, in_ws, allow_failure):
        out_mantid, time_mantid = self._execute_with_timing(self._run_mantid,
                                                            input=in_ws)
        in_da = mantid.from_mantid(in_ws)
        if in_da.data.bins is not None:
            in_da = in_da.astype(
                sc.dtype.float64)  # Converters set weights float32
        out_scipp, time_scipp = self._execute_with_timing(self._run_scipp,
                                                          input=in_da)

        self._assert(out_scipp, out_mantid, allow_failure)

        return {'scipp': out_scipp, 'mantid': out_mantid}

    def _append_result(self, name, result, results):
        results[f'with_{name}' if self._test_description is None else
                f'{self._test_description}_with_{name}'] = result

    def run(self, allow_failure=False):
        import mantid.simpleapi as sapi
        results = {}
        if self._filenames == {} and self._workspaces == {}:
            raise RuntimeError(
                'No _files or _workspaces provided for testing ')
        for name, (hash, algorithm) in self._filenames.items():
            file = MantidDataHelper.find_file(hash, algorithm)
            print('Loading', name)
            in_ws = sapi.Load(Filename=file, StoreInADS=False)
            result = self._run_from_workspace(in_ws, allow_failure)
            self._append_result(name, result, results)
        for name, in_ws in self._workspaces.items():
            result = self._run_from_workspace(in_ws, allow_failure)
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
