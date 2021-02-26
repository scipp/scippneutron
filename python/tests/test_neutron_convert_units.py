from mantid_scipp_comparison import MantidScippComparison
from mantid_data_helper import mantid_is_available
import pytest
import scippneutron.mantid as converter
import scippneutron as sn


class NeutronConvertUnitsTest(MantidScippComparison):
    @property
    def _dim_map(self):
        return {
            'tof': 'TOF',
            'wavelength': 'Wavelength',
            'd-spacing': 'dSpacing',
            'energy-transfer': 'DeltaE'
        }

    def __init__(self, name, origin, target, emode, efixed):
        self._origin = origin
        self._target = target
        self._emode = emode
        self._efixed = efixed
        super(NeutronConvertUnitsTest, self).__init__(name)

    @property
    def _workspaces(self):
        import mantid.simpleapi as sapi
        ws = sapi.CreateSampleWorkspace(NumBanks=1, StoreInADS=False)
        ws = sapi.ConvertUnits(InputWorkspace=ws,
                               Target=self._dim_map[self._origin],
                               EMode=self._emode,
                               EFixed=self._efixed,
                               StoreInADS=False)  # start in origin units
        return {"sample_workspace": ws}

    def _run_mantid(self, input):
        import mantid.simpleapi as sapi
        out = sapi.ConvertUnits(InputWorkspace=input,
                                Target=self._dim_map[self._target],
                                EMode=self._emode,
                                StoreInADS=False)
        return converter.from_mantid(out)

    def _run_scipp(self, input):
        return sn.convert(data=input, origin=self._origin, target=self._target)


class ElasticNeutronConvertUnitsTest(NeutronConvertUnitsTest):
    def __init__(self, origin, target):
        self._origin = origin
        self._target = target
        super(ElasticNeutronConvertUnitsTest,
              self).__init__(self.__class__.__name__,
                             origin,
                             target,
                             emode='Elastic',
                             efixed=None)


@pytest.mark.skipif(not mantid_is_available(),
                    reason='Mantid framework is unavailable')
def test_neutron_convert_units_tof_to_wavelength():
    test = ElasticNeutronConvertUnitsTest(origin='tof', target='wavelength')
    print(test.run(allow_failure=True))


@pytest.mark.skipif(not mantid_is_available(),
                    reason='Mantid framework is unavailable')
def test_neutron_convert_units_wavelength_to_tof():
    test = ElasticNeutronConvertUnitsTest(origin='wavelength', target='tof')
    print(test.run(allow_failure=True))


@pytest.mark.skipif(not mantid_is_available(),
                    reason='Mantid framework is unavailable')
def test_neutron_convert_units_tof_to_d_space():
    test = ElasticNeutronConvertUnitsTest(origin='tof', target='d-spacing')
    print(test.run(allow_failure=True))
