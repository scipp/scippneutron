from .mantid_scipp_comparison import MantidScippComparison
from ..mantid_helper import mantid_is_available
import pytest
import scippneutron.mantid as converter
import scipp as sc
import numpy as np


class HistogramEventsTest(MantidScippComparison):
    def __init__(self):
        super(HistogramEventsTest, self).__init__(self.__class__.__name__)

    @property
    def _filenames(self):
        return ["CNCS_51936_event.nxs"]

    def _run_mantid(self, input):
        import mantid.simpleapi as sapi
        # Note Mantid rebin inclusive of last bin boundary
        out = sapi.Rebin(InputWorkspace=input,
                         Params=[0, 10, 1000],
                         PreserveEvents=False,
                         StoreInADS=False)
        return converter.from_mantid(out)

    def _run_scipp(self, input):
        return sc.histogram(x=input,
                            bins=sc.Variable(dims=['tof'],
                                             values=np.linspace(0, 1000, num=101),
                                             dtype=sc.dtype.float64,
                                             unit=sc.units.us))


@pytest.mark.skipif(not mantid_is_available(), reason='Mantid framework is unavailable')
def test_histogram_events():
    test = HistogramEventsTest()
    test.run()
