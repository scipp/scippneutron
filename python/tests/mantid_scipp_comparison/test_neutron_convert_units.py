from .mantid_scipp_comparison import MantidScippComparison
from ..mantid_data_helper import mantid_is_available
import pytest
import scippneutron.mantid as converter
import scippneutron as sn
import scipp as sc
import enum


class Emode(enum.Enum):
    Elastic = 0
    Direct = 1


class Comparison(MantidScippComparison):
    @property
    def _dim_map(self):
        return {
            'tof': 'TOF',
            'wavelength': 'Wavelength',
            'dspacing': 'dSpacing',
            'energy_transfer': 'DeltaE'
        }

    @property
    def _emode_to_mantid(self):
        return {
            Emode.Elastic: 'Elastic',
            Emode.Direct: 'Direct',
        }

    def __init__(self, name, origin, target, emode, efixed):
        self._origin = origin
        self._target = target
        self._emode = emode
        self._efixed = efixed
        super(Comparison, self).__init__(name)

    @property
    def _workspaces(self):
        import mantid.simpleapi as sapi
        import mantid.kernel as kernel
        ws = sapi.CreateSampleWorkspace(XMin=1000, NumBanks=1, StoreInADS=False)
        # Crop out spectra index 0 as has two_theta=0, gives inf d-spacing
        ws = sapi.CropWorkspace(ws, StartWorkspaceIndex=1, StoreInADS=False)
        ws = sapi.ConvertUnits(
            InputWorkspace=ws,
            Target=self._dim_map[self._origin],
            EMode=self._emode_to_mantid[self._emode],
            EFixed=self._efixed.value if self._efixed is not None else None,
            StoreInADS=False)  # start in origin units
        ws.mutableRun().addProperty(
            'deltaE-mode',
            kernel.StringPropertyWithValue('deltaE-mode',
                                           self._emode_to_mantid[self._emode]), '',
            True)
        if self._efixed is not None:
            ws.mutableRun().addProperty(
                'Ei', kernel.FloatPropertyWithValue('Ei', self._efixed.value),
                str(self._efixed.unit), False)
        return {"sample_workspace": ws}

    def _run_mantid(self, input):
        import mantid.simpleapi as sapi
        out = sapi.ConvertUnits(
            InputWorkspace=input,
            Target=self._dim_map[self._target],
            EMode=self._emode_to_mantid[self._emode],
            EFixed=self._efixed.value if self._efixed is not None else None,
            StoreInADS=False)
        out = converter.from_mantid(out)
        # broadcast to circumvent common-bins conversion in from_mantid
        spec_shape = out.coords['spectrum'].shape
        out.coords[self._target] = sc.ones(dims=['spectrum'],
                                           shape=spec_shape) * out.coords[self._target]
        return out

    def _run_scipp(self, input):
        return sn.convert(data=input,
                          origin=self._origin,
                          target=self._target,
                          scatter=True)


class ElasticComparison(Comparison):
    def __init__(self, origin, target):
        self._origin = origin
        self._target = target
        super(ElasticComparison, self).__init__(self.__class__.__name__,
                                                origin,
                                                target,
                                                emode=Emode.Elastic,
                                                efixed=None)


class DirectInElasticComparison(Comparison):
    def __init__(self, origin, target):
        self._origin = origin
        self._target = target
        super(DirectInElasticComparison, self).__init__(self.__class__.__name__,
                                                        origin,
                                                        target,
                                                        emode=Emode.Direct,
                                                        efixed=1000 * sc.Unit('meV'))


@pytest.mark.skipif(not mantid_is_available(), reason='Mantid framework is unavailable')
class TestNeutronConversionUnits:
    def test_neutron_convert_units_tof_to_wavelength(self):
        test = ElasticComparison(origin='tof', target='wavelength')
        test.run()

    def test_neutron_convert_units_wavelength_to_tof(self):
        test = ElasticComparison(origin='wavelength', target='tof')
        test.run()

    def test_neutron_convert_units_tof_to_d_space(self):
        test = ElasticComparison(origin='tof', target='dspacing')
        test.run()

    def test_neutron_convert_units_tof_to_wavelength_direct(self):
        test = DirectInElasticComparison(origin='tof', target='energy_transfer')
        test.run()
