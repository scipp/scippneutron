from .mantid_scipp_comparison import MantidScippComparison
from ..mantid_data_helper import mantid_is_available
import pytest
import scippneutron as sn
import scipp as sc


class BeamlineCalculationsTest(MantidScippComparison):
    def __init__(self):
        super(BeamlineCalculationsTest, self).__init__(self.__class__.__name__)

    @property
    def _workspaces(self):
        import mantid.simpleapi as sapi
        ws = sapi.CreateSampleWorkspace(SourceDistanceFromSample=10.0,
                                        BankDistanceFromSample=1.1,
                                        BankPixelWidth=1,
                                        NumBanks=1,
                                        XMax=200,
                                        StoreInADS=False)
        return {"sample_workspace": ws}


class L1CalculationsTest(BeamlineCalculationsTest):
    def _run_mantid(self, input):
        return input.detectorInfo().l1() * sc.Unit('m')

    def _run_scipp(self, input):
        return sn.l1(input)


class L2CalculationsTest(BeamlineCalculationsTest):
    def _run_mantid(self, input):
        return input.detectorInfo().l2(0) * sc.Unit('m')

    def _run_scipp(self, input):
        return sn.l2(input)


class TwoThetaCalculationsTest(BeamlineCalculationsTest):
    def _run_mantid(self, input):
        return input.detectorInfo().twoTheta(0) * sc.Unit('rad')

    def _run_scipp(self, input):
        return sn.scattering_angle(input)


@pytest.mark.skipif(not mantid_is_available(),
                    reason='Mantid framework is unavailable')
def test_bealine_calculations_l1():
    test = L1CalculationsTest()
    print(test.run(allow_failure=True))


@pytest.mark.skipif(not mantid_is_available(),
                    reason='Mantid framework is unavailable')
def test_bealine_calculations_l2():
    test = L2CalculationsTest()
    print(test.run(allow_failure=True))


@pytest.mark.skipif(not mantid_is_available(),
                    reason='Mantid framework is unavailable')
def test_bealine_calculations_two_theta():
    test = TwoThetaCalculationsTest()
    print(test.run(allow_failure=True))
