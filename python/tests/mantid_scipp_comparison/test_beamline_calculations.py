from .mantid_scipp_comparison import MantidScippComparison
from ..mantid_data_helper import mantid_is_available
import pytest
import scippneutron.mantid as converter
import scippneutron as sn
import scipp as sc


class BeamlineCalculationsTest(MantidScippComparison):
    def __init__(self):
        super(BeamlineCalculationsTest, self).__init__(self.__class__.__name__)

    @property
    def _workspaces(self):
        import mantid.simpleapi as sapi
        import mantid.kernel as kernel
        ws = sapi.CreateSampleWorkspace(SourceDistanceFromSample=10.0,
                                        BankDistanceFromSample=1.1,
                                        NumBanks=1,
                                        StoreInADS=False)
        return {"sample_workspace": ws}

    def _run_mantid(self, input):
        import mantid.simpleapi as sapi
        return input.detectorInfo().l1() * sc.Unit('m')

    def _run_scipp(self, input):
        return sn.l1(input)


@pytest.mark.skipif(not mantid_is_available(),
                    reason='Mantid framework is unavailable')
def test_bealine_calculations_l1():
    test = BeamlineCalculationsTest()
    print(test.run(allow_failure=True))
