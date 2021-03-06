from .mantid_scipp_comparison import MantidScippComparison
from ..mantid_data_helper import mantid_is_available
import pytest
import scippneutron as sn
import scipp as sc


class BeamlineComparision(MantidScippComparison):
    def __init__(self):
        super(BeamlineComparision, self).__init__(self.__class__.__name__)

    @property
    def _workspaces(self):
        import mantid.simpleapi as sapi
        ws = sapi.CreateSampleWorkspace(SourceDistanceFromSample=10.0,
                                        BankDistanceFromSample=1.1,
                                        BankPixelWidth=2,
                                        NumBanks=1,
                                        XMax=200,
                                        StoreInADS=False)
        return {"sample_workspace": ws}


@pytest.mark.skipif(not mantid_is_available(),
                    reason='Mantid framework is unavailable')
class TestBeamlineCalculations:
    def test_beamline_calculations_l1(self):
        class L1Comparison(BeamlineComparision):
            def _run_mantid(self, data):
                return data.detectorInfo().l1() * sc.Unit('m')

            def _run_scipp(self, data):
                return sn.L1(data)

        test = L1Comparison()
        test.run()

    def test_bealine_calculations_l2(self):
        class L2Comparison(BeamlineComparision):
            def _run_mantid(self, data):
                return sc.array(
                    dims=['spectrum'],
                    unit='m',
                    values=[data.detectorInfo().l2(i) for i in [0, 1, 2, 3]])

            def _run_scipp(self, data):
                return sn.L2(data)

        test = L2Comparison()
        test.run()

    def test_bealine_calculations_two_theta(self):
        class TwoThetaComparison(BeamlineComparision):
            def _run_mantid(self, data):
                return sc.array(dims=['spectrum'],
                                unit='rad',
                                values=[
                                    data.detectorInfo().twoTheta(i)
                                    for i in [0, 1, 2, 3]
                                ])

            def _run_scipp(self, data):
                return sn.two_theta(data)

        test = TwoThetaComparison()
        test.run()
