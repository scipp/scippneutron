# Tests in this file work only with a working Mantid installation available in
# PYTHONPATH.
import importlib
import os
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pytest
import scipp as sc

import scippneutron as scn
from scippneutron._utils import get_attrs

mantid = pytest.importorskip(
    'mantid.simpleapi', reason='Mantid framework is unavailable'
)


def memory_is_at_least_gb(required):
    import psutil

    total = psutil.virtual_memory().total / 1e9
    return total >= required


@pytest.mark.filterwarnings("ignore:convert_Workspace2D_to_data_array")
@pytest.mark.filterwarnings("ignore:convert_EventWorkspace_to_data_array")
@pytest.mark.filterwarnings("ignore:convert_MDHistoWorkspace_to_data_array")
@pytest.mark.filterwarnings("ignore:scippneutron.array_from_mantid")
@pytest.mark.skipif(not memory_is_at_least_gb(4), reason='Insufficient virtual memory')
class TestMantidConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This is from the Mantid system-test data
        filename = "CNCS_51936_event.nxs"
        # This needs OutputWorkspace specified, as it doesn't
        # pick up the name from the class variable name
        cls.base_event_ws = mantid.LoadEventNexus(
            scn.data.get_path(filename),
            OutputWorkspace=f"test_ws{__file__}",
            SpectrumMax=200,
            StoreInADS=False,
        )

    def test_Workspace2D_data_array(self):
        eventWS = self.base_event_ws
        ws = mantid.Rebin(eventWS, 10000, PreserveEvents=False)
        d = scn.mantid.convert_Workspace2D_to_data_array(ws)
        assert get_attrs(d)["run_start"].value == "2012-05-21T15:14:56.279289666"
        assert d.data.unit == sc.units.counts
        for i in range(ws.getNumberHistograms()):
            assert np.all(np.equal(d.values[i], ws.readY(i)))
            assert np.all(np.equal(d.variances[i], ws.readE(i) * ws.readE(i)))
        assert d.coords['spectrum'].dtype == sc.DType.int32
        assert d.coords['spectrum'].unit is None
        assert d.coords['tof'].dtype == sc.DType.float64

    def test_Workspace2D(self):
        eventWS = self.base_event_ws
        ws = mantid.Rebin(eventWS, 10000, PreserveEvents=False)
        d = scn.mantid.convert_Workspace2D_to_data_group(ws)
        assert d["run_start"].value == "2012-05-21T15:14:56.279289666"
        assert d["data"].unit == sc.units.counts
        for i in range(ws.getNumberHistograms()):
            assert np.all(np.equal(d["data"].values[i], ws.readY(i)))
            assert np.all(np.equal(d["data"].variances[i], ws.readE(i) * ws.readE(i)))
        assert d["data"].coords['spectrum'].dtype == sc.DType.int32
        assert d["data"].coords['spectrum'].unit is None
        assert d["data"].coords['tof'].dtype == sc.DType.float64

    def test_EventWorkspace_data_array(self):
        eventWS = self.base_event_ws
        ws = mantid.Rebin(eventWS, 10000)

        binned_mantid = scn.mantid.convert_Workspace2D_to_data_array(ws)

        target_tof = binned_mantid.coords['tof']
        d = scn.mantid.convert_EventWorkspace_to_data_array(
            eventWS, load_pulse_times=False
        )
        histogrammed = d.hist(tof=target_tof)

        delta = sc.sum(binned_mantid - histogrammed, 'spectrum')
        delta = sc.sum(delta, 'tof')
        assert np.abs(delta.value) < 1e-05

    def test_EventWorkspace(self):
        eventWS = self.base_event_ws
        ws = mantid.Rebin(eventWS, 10000)

        binned_mantid = scn.mantid.convert_Workspace2D_to_data_group(ws)["data"]

        target_tof = binned_mantid.coords['tof']
        d = scn.mantid.convert_EventWorkspace_to_data_group(
            eventWS, load_pulse_times=False
        )
        assert d["run_start"].value == "2012-05-21T15:14:56.279289666"

        histogrammed = d["data"].hist(tof=target_tof)
        delta = sc.sum(binned_mantid - histogrammed, 'spectrum')
        delta = sc.sum(delta, 'tof')
        assert np.abs(delta.value) < 1e-5

    def test_EventWorkspace_empty_event_list_consistent_bin_indices(self):
        ws = mantid.CloneWorkspace(self.base_event_ws)
        ws.getSpectrum(ws.getNumberHistograms() - 1).clear(removeDetIDs=True)

        dg = scn.mantid.convert_EventWorkspace_to_data_group(ws, load_pulse_times=False)
        assert dg["run_start"].value == "2012-05-21T15:14:56.279289666"
        da = dg["data"]
        assert da.bins.size()['spectrum', -1]['tof', 0].value == 0

    def test_comparison(self):
        a = scn.mantid.convert_EventWorkspace_to_data_group(
            self.base_event_ws, load_pulse_times=False
        )
        b = a.copy()
        assert sc.identical(a, b)

    def test_advanced_geometry(self):
        # basic test that positions are approximately equal for detectors for
        # CNCS given advanced and basic geometry calculation routes
        x = scn.from_mantid(self.base_event_ws, advanced_geometry=False)
        y = scn.from_mantid(self.base_event_ws, advanced_geometry=True)
        assert sc.allclose(x["data"].coords['position'], y["data"].coords['position'])

    def test_advanced_geometry_with_absent_shape(self):
        # single bank 3 by 3
        ws = mantid.CreateSampleWorkspace(
            NumBanks=1, BankPixelWidth=3, StoreInADS=False
        )
        # Save and reload trick to purge sample shape info
        file_name = "example_geometry.nxs"
        geom_path = os.path.join(tempfile.gettempdir(), file_name)
        mantid.SaveNexusGeometry(ws, geom_path)  # Does not save shape info
        assert os.path.isfile(geom_path)  # sanity check
        out = mantid.LoadEmptyInstrument(
            Filename=geom_path, StoreInADS=False
        )  # reload without sample info
        os.remove(geom_path)

        assert not out.componentInfo().hasValidShape(0)  # sanity check
        dg = scn.mantid.from_mantid(out, advanced_geometry=True)
        # Shapes have zero size
        assert sc.identical(
            sc.sum(dg['shape']), sc.vector(value=[0, 0, 0], unit=sc.units.m)
        )

    def test_advanced_geometry_detector_info(self):
        dg = scn.from_mantid(self.base_event_ws, advanced_geometry=True)
        detector_info = dg['data'].coords['detector_info'].value
        assert detector_info.dim == 'detector'
        assert detector_info.coords['detector'].unit is None
        assert detector_info.coords['spectrum'].unit is None

    def test_EventWorkspace_no_y_unit(self):
        tiny_event_ws = mantid.CreateSampleWorkspace(
            WorkspaceType='Event', NumBanks=1, NumEvents=1
        )
        d = scn.mantid.convert_EventWorkspace_to_data_array(
            tiny_event_ws, load_pulse_times=False
        )
        assert d.data.bins.constituents['data'].unit == sc.units.counts
        tiny_event_ws.setYUnit('')
        d = scn.mantid.convert_EventWorkspace_to_data_array(
            tiny_event_ws, load_pulse_times=False
        )
        assert d.data.bins.constituents['data'].unit == sc.units.one

    def test_from_mantid_LoadEmptyInstrument(self):
        ws = mantid.LoadEmptyInstrument(InstrumentName='PG3')
        scn.from_mantid(ws)

    def test_from_mantid_CreateWorkspace(self):
        dataX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        dataY = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        ws = mantid.CreateWorkspace(
            DataX=dataX, DataY=dataY, NSpec=4, UnitX="Wavelength"
        )
        d = scn.from_mantid(ws)
        assert d['data'].unit == sc.units.dimensionless

    def test_unit_conversion(self):
        eventWS = self.base_event_ws
        ws = mantid.Rebin(eventWS, 10000, PreserveEvents=False)
        tmp = scn.mantid.convert_Workspace2D_to_data_array(ws)
        target_tof = tmp.coords['tof']
        ws = mantid.ConvertUnits(
            InputWorkspace=ws, Target="Wavelength", EMode="Elastic"
        )
        converted_mantid = scn.mantid.convert_Workspace2D_to_data_group(ws)

        da = scn.mantid.convert_EventWorkspace_to_data_array(
            eventWS, load_pulse_times=False
        )
        da = da.hist(tof=target_tof)
        d = sc.Dataset(data={da.name: da})
        converted = scn.convert(d, 'tof', 'wavelength', scatter=True)

        assert sc.allclose(
            converted_mantid['data'].data, converted[""].data.to(dtype='float64')
        )
        assert sc.allclose(
            converted_mantid['data'].coords['wavelength'],
            converted.coords['wavelength'],
        )

    def test_inelastic_unit_conversion(self):
        eventWS = self.base_event_ws
        ws_deltaE = mantid.ConvertUnits(
            eventWS, Target='DeltaE', EMode='Direct', EFixed=3
        )
        ref = scn.from_mantid(ws_deltaE)['data']
        da = scn.from_mantid(eventWS)['data']
        # Boost and Mantid use CODATA 2006. This test passes if we manually
        # change the implementation to use the old constants. Alternatively
        # we can correct for this by scaling L1^2 or L2^2, and this was also
        # confirmed in C++. Unfortunately only positions are accessible to
        # correct for this here, and due to precision issues with
        # dot/norm/sqrt this doesn't actually fix the test. We additionally
        # exclude low TOF region, and bump relative and absolute accepted
        # errors from 1e-8 to 1e-5.
        m_n_2006 = 1.674927211
        m_n_2018 = 1.67492749804
        e_2006 = 1.602176487
        e_2018 = 1.602176634
        scale = (m_n_2006 / m_n_2018) / (e_2006 / e_2018)
        da.coords['source_position'] *= np.sqrt(scale)
        da.coords['position'] *= np.sqrt(scale)
        low_tof = da.bins.constituents['data'].coords['tof'] < 49000.0 * sc.units.us
        da.coords['incident_energy'] = 3.0 * sc.units.meV
        da = scn.convert(da, 'tof', 'energy_transfer', scatter=True)
        assert sc.all(
            sc.isnan(da.coords['energy_transfer'])
            | sc.isclose(
                da.coords['energy_transfer'],
                ref.coords['energy_transfer'],
                atol=1e-8 * sc.units.meV,
                rtol=1e-8 * sc.units.one,
            )
        ).value
        assert sc.all(
            low_tof
            | sc.isnan(da.bins.constituents['data'].coords['energy_transfer'])
            | sc.isclose(
                da.bins.constituents['data'].coords['energy_transfer'],
                ref.bins.constituents['data'].coords['energy_transfer'],
                atol=1e-5 * sc.units.meV,
                rtol=1e-5 * sc.units.one,
            )
        ).value

    @staticmethod
    def _mask_bins_and_spectra(ws, xmin, xmax, num_spectra, indices=None):
        masked_ws = mantid.MaskBins(
            ws, XMin=xmin, XMax=xmax, InputWorkspaceIndexSet=indices
        )

        # mask the first 3 spectra
        for i in range(num_spectra):
            masked_ws.spectrumInfo().setMasked(i, True)

        return masked_ws

    def test_Workspace2D_common_bins_masks(self):
        eventWS = self.base_event_ws
        ws = mantid.Rebin(eventWS, 10000, PreserveEvents=False)
        ws_x = ws.readX(0)

        # mask the first 3 bins, range is taken as [XMin, XMax)
        masked_ws = self._mask_bins_and_spectra(
            ws, xmin=ws_x[0], xmax=ws_x[3], num_spectra=3
        )

        assert masked_ws.isCommonBins()

        da = scn.mantid.convert_Workspace2D_to_data_group(masked_ws)['data']

        np.testing.assert_array_equal(da.masks["bin"].values[0:3], [True, True, True])

        np.testing.assert_array_equal(
            da.masks["spectrum"].values[0:3], [True, True, True]
        )

    def test_Workspace2D_common_bins_not_common_masks(self):
        eventWS = self.base_event_ws
        ws = mantid.Rebin(eventWS, 10000, PreserveEvents=False)
        ws_x = ws.readX(0)

        # mask first 3 bins in first 3 spectra, range is taken as [XMin, XMax)
        masked_ws = self._mask_bins_and_spectra(
            ws, xmin=ws_x[0], xmax=ws_x[3], num_spectra=3, indices='0-2'
        )

        assert masked_ws.isCommonBins()

        da = scn.mantid.convert_Workspace2D_to_data_group(masked_ws)['data']

        mask = sc.zeros(dims=da.dims, shape=da.shape, dtype=sc.DType.bool)
        mask['spectrum', 0:3]['tof', 0:3] |= sc.scalar(True)
        assert sc.identical(da.masks['bin'], mask)

        np.testing.assert_array_equal(
            da.masks["spectrum"].values[0:3], [True, True, True]
        )

    def test_Workspace2D_not_common_bins_masks(self):
        eventWS = self.base_event_ws
        ws = mantid.Rebin(eventWS, 10000, PreserveEvents=False)
        ws = mantid.ConvertUnits(ws, "Wavelength", EMode="Direct", EFixed=0.1231)

        # these X values will mask different number of bins
        masked_ws = self._mask_bins_and_spectra(
            ws, -214, -192, num_spectra=3, indices='0-40'
        )

        assert not masked_ws.isCommonBins()

        da = scn.mantid.convert_Workspace2D_to_data_group(masked_ws)['data']

        # bin with 3 masks
        np.testing.assert_array_equal(da.masks["bin"].values[0], [True, True, False])

        # bin with only 2
        np.testing.assert_array_equal(da.masks["bin"].values[31], [True, True, False])

        np.testing.assert_array_equal(
            da.masks["spectrum"].values[0:3], [True, True, True]
        )

    @staticmethod
    def check_monitor_metadata(monitor):
        assert 'position' in monitor.coords
        assert 'source_position' in monitor.coords
        assert not monitor.coords['sample_position'].aligned
        # Absence of the following is not crucial, but currently there is
        # no need for these, and it avoids duplication:
        assert 'detector_info' not in monitor.coords
        assert 'sample' not in monitor.coords
        assert (
            'SampleTemp' not in monitor.coords
        ), "Expect run logs not be duplicated in monitor workspaces"

    @staticmethod
    def check_monitor_metadata_old(monitor):
        # TODO remove once load and array_from_mantid are removed
        assert 'position' in monitor.coords
        assert 'source_position' in monitor.coords
        assert 'sample_position' not in monitor.coords
        assert 'sample_position' in get_attrs(monitor)
        # Absence of the following is not crucial, but currently there is
        # no need for these, and it avoids duplication:
        assert 'detector_info' not in monitor.coords
        assert 'sample' not in monitor.coords
        assert (
            'SampleTemp' not in monitor.coords
        ), "Expect run logs not be duplicated in monitor workspaces"

    def test_Workspace2D_with_separate_monitors(self):
        from mantid.simpleapi import mtd

        mtd.clear()
        # This test would use 20 GB of memory if "SpectrumMax" was not set
        dg = scn.load_with_mantid(
            scn.data.get_path("WISH00016748.raw"),
            mantid_args={"LoadMonitors": "Separate", "SpectrumMax": 10000},
        )
        assert len(mtd) == 0, mtd.getObjectNames()
        expected_monitor_names = {
            "monitor1",
            "monitor2",
            "monitor3",
            "monitor4",
            "monitor5",
        }
        assert dg["monitors"].keys() == expected_monitor_names

        for monitor in dg["monitors"].values():
            assert isinstance(monitor, sc.DataGroup)
            assert monitor.shape == (4471,)
            self.check_monitor_metadata(monitor['data'])

    @pytest.mark.filterwarnings("ignore:scippneutron.load")
    def test_load_Workspace2D_with_separate_monitors(self):
        from mantid.simpleapi import mtd

        mtd.clear()
        # This test would use 20 GB of memory if "SpectrumMax" was not set
        da = scn.load(
            scn.data.get_path("WISH00016748.raw"),
            mantid_args={"LoadMonitors": "Separate", "SpectrumMax": 10000},
        )
        assert len(mtd) == 0, mtd.getObjectNames()
        expected_monitor_names = {
            "monitor1",
            "monitor2",
            "monitor3",
            "monitor4",
            "monitor5",
        }
        assert expected_monitor_names.issubset(get_attrs(da).keys())

        for monitor_name in expected_monitor_names:
            monitor = get_attrs(da)[monitor_name].value
            assert isinstance(monitor, sc.DataArray)
            assert monitor.shape == (4471,)
            self.check_monitor_metadata_old(monitor)

    def test_Workspace2D_with_include_monitors(self):
        from mantid.simpleapi import mtd

        mtd.clear()
        # This test would use 20 GB of memory if "SpectrumMax" was not set
        dg = scn.load_with_mantid(
            scn.data.get_path("WISH00016748.raw"),
            mantid_args={"LoadMonitors": "Include", "SpectrumMax": 100},
        )
        assert len(mtd) == 0, mtd.getObjectNames()
        expected_monitor_names = {
            "monitor1",
            "monitor2",
            "monitor3",
            "monitor4",
            "monitor5",
        }
        assert dg["monitors"].keys() == expected_monitor_names
        for monitor in dg["monitors"].values():
            assert isinstance(monitor, sc.DataGroup)
            assert monitor.shape == (4471,)
            self.check_monitor_metadata(monitor['data'])

    @pytest.mark.filterwarnings("ignore:scippneutron.load")
    def test_load_Workspace2D_with_include_monitors(self):
        from mantid.simpleapi import mtd

        mtd.clear()
        # This test would use 20 GB of memory if "SpectrumMax" was not set
        da = scn.load(
            scn.data.get_path("WISH00016748.raw"),
            mantid_args={"LoadMonitors": "Include", "SpectrumMax": 100},
        )
        assert len(mtd) == 0, mtd.getObjectNames()
        expected_monitor_names = {
            "monitor1",
            "monitor2",
            "monitor3",
            "monitor4",
            "monitor5",
        }
        assert expected_monitor_names.issubset(get_attrs(da).keys())
        for monitor_name in expected_monitor_names:
            monitor = get_attrs(da)[monitor_name].value
            assert isinstance(monitor, sc.DataArray)
            assert monitor.shape == (4471,)
            self.check_monitor_metadata_old(monitor)

    def test_EventWorkspace_with_monitors(self):
        from mantid.simpleapi import mtd

        mtd.clear()
        dg = scn.load_with_mantid(
            scn.data.get_path("CNCS_51936_event.nxs"),
            mantid_args={"LoadMonitors": True, "SpectrumMax": 1},
        )
        assert len(mtd) == 0, mtd.getObjectNames()
        expected_monitor_attrs = {"monitor2", "monitor3"}
        assert dg["monitors"].keys() == expected_monitor_attrs
        for monitor in dg["monitors"].values():
            assert isinstance(monitor, sc.DataGroup)
            assert monitor.shape == (200001,)
            self.check_monitor_metadata(monitor['data'])

    @pytest.mark.filterwarnings("ignore:scippneutron.load")
    def test_load_EventWorkspace_with_monitors(self):
        from mantid.simpleapi import mtd

        mtd.clear()
        da = scn.load(
            scn.data.get_path("CNCS_51936_event.nxs"),
            mantid_args={"LoadMonitors": True, "SpectrumMax": 1},
        )
        assert len(mtd) == 0, mtd.getObjectNames()
        attrs = [str(key) for key in get_attrs(da).keys()]
        expected_monitor_attrs = {"monitor2", "monitor3"}
        assert expected_monitor_attrs.issubset(attrs)
        for monitor_name in expected_monitor_attrs:
            monitor = get_attrs(da)[monitor_name].value
            assert isinstance(monitor, sc.DataArray)
            assert monitor.shape == (200001,)
            self.check_monitor_metadata_old(monitor)

    def test_mdhisto_workspace_q(self):
        from mantid.simpleapi import BinMD, CreateMDWorkspace, FakeMDEventData

        md_event = CreateMDWorkspace(
            Dimensions=3,
            Extents=[-10, 10, -10, 10, -10, 10],
            Names='Q_x,Q_y,Q_z',
            Units='U,U,U',
            Frames='QLab,QLab,QLab',
            StoreInADS=False,
        )
        FakeMDEventData(
            InputWorkspace=md_event, PeakParams=[100000, 0, 0, 0, 1], StoreInADS=False
        )  # Add Peak
        md_histo = BinMD(
            InputWorkspace=md_event,
            AlignedDim0='Q_y,-10,10,3',
            AlignedDim1='Q_x,-10,10,4',
            AlignedDim2='Q_z,-10,10,5',
            StoreInADS=False,
        )

        histo_data_group = scn.mantid.convert_MDHistoWorkspace_to_data_group(md_histo)
        histo_data_array = histo_data_group['data']

        assert histo_data_array.coords['Q_x'].shape == (4,)
        assert histo_data_array.coords['Q_y'].shape == (3,)
        assert histo_data_array.coords['Q_z'].shape == (5,)
        assert (
            histo_data_array.coords['Q_x'].unit
            == sc.units.dimensionless / sc.units.angstrom
        )
        assert (
            histo_data_array.coords['Q_y'].unit
            == sc.units.dimensionless / sc.units.angstrom
        )
        assert (
            histo_data_array.coords['Q_z'].unit
            == sc.units.dimensionless / sc.units.angstrom
        )
        assert histo_data_array.shape == (3, 4, 5)

        # Sum over 2 dimensions to simplify finding max.
        max_1d = sc.sum(sc.sum(histo_data_array, dim='Q_y'), dim='Q_x').values
        max_index = np.argmax(max_1d)
        # Check position of max 'peak'
        assert np.floor(len(max_1d) / 2) == max_index
        # All events in central 'peak'
        assert 100000 == max_1d[max_index]

        assert 'nevents' in histo_data_group

    def test_mdhisto_workspace_q_data_array(self):
        from mantid.simpleapi import BinMD, CreateMDWorkspace, FakeMDEventData

        md_event = CreateMDWorkspace(
            Dimensions=3,
            Extents=[-10, 10, -10, 10, -10, 10],
            Names='Q_x,Q_y,Q_z',
            Units='U,U,U',
            Frames='QLab,QLab,QLab',
            StoreInADS=False,
        )
        FakeMDEventData(
            InputWorkspace=md_event, PeakParams=[100000, 0, 0, 0, 1], StoreInADS=False
        )  # Add Peak
        md_histo = BinMD(
            InputWorkspace=md_event,
            AlignedDim0='Q_y,-10,10,3',
            AlignedDim1='Q_x,-10,10,4',
            AlignedDim2='Q_z,-10,10,5',
            StoreInADS=False,
        )

        histo_data_array = scn.mantid.convert_MDHistoWorkspace_to_data_array(md_histo)

        assert histo_data_array.coords['Q_x'].shape == (4,)
        assert histo_data_array.coords['Q_y'].shape == (3,)
        assert histo_data_array.coords['Q_z'].shape == (5,)
        assert (
            histo_data_array.coords['Q_x'].unit
            == sc.units.dimensionless / sc.units.angstrom
        )
        assert (
            histo_data_array.coords['Q_y'].unit
            == sc.units.dimensionless / sc.units.angstrom
        )
        assert (
            histo_data_array.coords['Q_z'].unit
            == sc.units.dimensionless / sc.units.angstrom
        )

        assert histo_data_array.shape == (3, 4, 5)

        # Sum over 2 dimensions to simplify finding max.
        max_1d = sc.sum(sc.sum(histo_data_array, dim='Q_y'), dim='Q_x').values
        max_index = np.argmax(max_1d)
        # Check position of max 'peak'
        assert np.floor(len(max_1d) / 2) == max_index
        # All events in central 'peak'
        assert 100000 == max_1d[max_index]

        assert 'nevents' in get_attrs(histo_data_array)

    def test_mdhisto_workspace_many_dims(self):
        from mantid.simpleapi import BinMD, CreateMDWorkspace, FakeMDEventData

        md_event = CreateMDWorkspace(
            Dimensions=4,
            Extents=[-10, 10, -10, 10, -10, 10, -10, 10],
            Names='deltae,y,z,T',
            Units='U,U,U,U',
            StoreInADS=False,
        )
        FakeMDEventData(
            InputWorkspace=md_event,
            PeakParams=[100000, 0, 0, 0, 0, 1],
            StoreInADS=False,
        )  # Add Peak
        md_histo = BinMD(
            InputWorkspace=md_event,
            AlignedDim0='deltae,-10,10,3',
            AlignedDim1='y,-10,10,4',
            AlignedDim2='z,-10,10,5',
            AlignedDim3='T,-10,10,7',
            StoreInADS=False,
        )

        histo_data_group = scn.mantid.convert_MDHistoWorkspace_to_data_group(md_histo)
        assert len(histo_data_group.dims) == 4

    def test_mdhisto_workspace_many_dims_array(self):
        from mantid.simpleapi import BinMD, CreateMDWorkspace, FakeMDEventData

        md_event = CreateMDWorkspace(
            Dimensions=4,
            Extents=[-10, 10, -10, 10, -10, 10, -10, 10],
            Names='deltae,y,z,T',
            Units='U,U,U,U',
            StoreInADS=False,
        )
        FakeMDEventData(
            InputWorkspace=md_event,
            PeakParams=[100000, 0, 0, 0, 0, 1],
            StoreInADS=False,
        )  # Add Peak
        md_histo = BinMD(
            InputWorkspace=md_event,
            AlignedDim0='deltae,-10,10,3',
            AlignedDim1='y,-10,10,4',
            AlignedDim2='z,-10,10,5',
            AlignedDim3='T,-10,10,7',
            StoreInADS=False,
        )

        histo_data_array = scn.mantid.convert_MDHistoWorkspace_to_data_array(md_histo)
        assert 4 == len(histo_data_array.dims)

    def test_to_workspace_2d_no_error(self):
        from mantid.simpleapi import mtd

        mtd.clear()

        # All Dims for which support is expected are
        # tested in the parametrized test.
        # Just set this one to a working one to avoid
        # generating many repetitive tests.
        param_dim = 'tof'
        data_len = 2
        expected_bins = data_len + 1
        expected_number_spectra = 10

        y = sc.Variable(
            dims=['spectrum', param_dim],
            values=np.random.rand(expected_number_spectra, data_len),
        )

        x = sc.Variable(
            dims=['spectrum', param_dim],
            values=np.arange(
                expected_number_spectra * expected_bins, dtype=np.float64
            ).reshape((expected_number_spectra, expected_bins)),
        )
        data = sc.DataArray(data=y, coords={param_dim: x})

        ws = scn.to_mantid(data, param_dim)

        assert len(ws.readX(0)) == expected_bins
        assert ws.getNumberHistograms() == expected_number_spectra
        # check that no workspaces have been leaked in the ADS
        assert len(mtd) == 0, f"Workspaces present: {mtd.getObjectNames()}"

        for i in range(expected_number_spectra):
            np.testing.assert_array_equal(ws.readX(i), x['spectrum', i].values)
            np.testing.assert_array_equal(ws.readY(i), y['spectrum', i].values)
            np.testing.assert_array_equal(ws.readE(i), np.sqrt(y['spectrum', i].values))

    def test_fit(self):
        """
        Tests that the fit executes, and the outputs
        are moved into the dataset. Does not check the fit values.
        """
        from mantid.simpleapi import mtd

        mtd.clear()

        data = scn.load_with_mantid(scn.data.get_path("iris26176_graphite002_sqw.nxs"))

        params, diff = scn.fit(
            data['Q', 0]['data'],
            mantid_args={
                'Function': 'name=LinearBackground,A0=0,A1=1',
                'StartX': 0,
                'EndX': 3,
            },
        )

        # check that no workspaces have been leaked in the ADS
        assert len(mtd) == 0
        assert 'data' in diff
        assert 'calculated' in diff
        assert 'diff' in diff
        assert 'status' in params.coords
        assert 'function' in params.coords
        assert 'cost_function' in params.coords
        assert 'chi^2/d.o.f.' in params.coords

    def test_convert_array_run_log_to_attrs(self):
        # Given a Mantid workspace with a run log
        target = mantid.CloneWorkspace(self.base_event_ws)
        log_name = "SampleTemp"
        assert target.run().hasProperty(
            log_name
        ), f'Expected input workspace to have a {log_name} run log'

        # When the workspace is converted to a scipp data array
        d = scn.mantid.convert_EventWorkspace_to_data_array(target, False)

        # Then the data array contains the run log as an unaligned coord
        assert np.allclose(
            target.run()[log_name].value, get_attrs(d)[log_name].values.data.values
        ), (
            'Expected values in the unaligned coord to match '
            'the original run log from the Mantid workspace'
        )
        assert get_attrs(d)[log_name].values.unit == sc.units.K
        assert np.array_equal(
            target.run()[log_name].times.astype('datetime64[ns]'),
            get_attrs(d)[log_name].values.coords['time'].values,
        ), (
            'Expected times in the unaligned coord to match '
            'the original run log from the Mantid workspace'
        )

    def test_convert_scalar_run_log_to_attrs(self):
        # Given a Mantid workspace with a run log
        target = mantid.CloneWorkspace(self.base_event_ws)
        log_name = "start_time"
        assert target.run().hasProperty(
            log_name
        ), f'Expected input workspace to have a {log_name} run log'

        # When the workspace is converted to a scipp data array
        d = scn.mantid.convert_EventWorkspace_to_data_array(target, False)

        # Then the data array contains the run log as an unaligned coord
        assert target.run()[log_name].value == get_attrs(d)[log_name].value, (
            'Expected value of the unaligned coord to match '
            'the original run log from the Mantid workspace'
        )

    def test_warning_raised_when_convert_run_log_with_unrecognised_units(self):
        target = mantid.CloneWorkspace(self.base_event_ws)
        target.getRun()['LambdaRequest'].units = 'abcde'
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            scn.mantid.convert_EventWorkspace_to_data_array(target, False)
            assert len(caught_warnings) > 0, (
                "Expected warnings due to some run logs "
                "having unrecognised units strings"
            )
            assert any(
                "unrecognised units" in str(caught_warning.message)
                for caught_warning in caught_warnings
            )

    def test_no_warning_raised_explicitly_dimensionless_run_log(self):
        target = mantid.CloneWorkspace(self.base_event_ws)
        with warnings.catch_warnings(record=True) as caught_warnings:
            scn.mantid.convert_EventWorkspace_to_data_array(target, False)
            original_number_of_warnings = len(caught_warnings)

        # Add an explicitly dimensionless log
        mantid.AddSampleLog(
            Workspace=target,
            LogName='dimensionless_log',
            LogText='1',
            LogType='Number',
            LogUnit='dimensionless',
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            scn.mantid.convert_EventWorkspace_to_data_array(target, False)
            assert len(caught_warnings) == original_number_of_warnings, (
                "Expected no extra warning about unrecognised units "
                "from explicitly dimensionless log"
            )

    def test_set_sample(self):
        target = mantid.CloneWorkspace(self.base_event_ws)
        d = scn.mantid.convert_EventWorkspace_to_data_array(target, False)
        get_attrs(d)["sample"].value.setThickness(3)
        # before
        assert 3 != target.sample().getThickness()
        target.setSample(get_attrs(d)["sample"].value)
        # after
        assert 3 == target.sample().getThickness()

    def test_sample_ub(self):
        ws = mantid.CreateWorkspace(DataY=np.ones(1), DataX=np.arange(2))
        args = {'a': 1, 'b': 1, 'c': 1, 'alpha': 90, 'beta': 90, 'gamma': 90}
        mantid.SetUB(ws, **args)
        d = scn.mantid.from_mantid(ws)
        assert sc.identical(
            d['sample_ub'],
            sc.spatial.linear_transform(
                value=ws.sample().getOrientedLattice().getUB(),
                unit=sc.units.angstrom**-1,
            ),
        )
        assert sc.identical(
            d['sample_u'],
            sc.spatial.linear_transform(value=ws.sample().getOrientedLattice().getU()),
        )

    def test_sample_without_ub(self):
        ws = mantid.CreateWorkspace(DataY=np.ones(1), DataX=np.arange(2))
        assert not ws.sample().hasOrientedLattice()  # Sanity check input
        d = scn.mantid.from_mantid(ws)
        assert "sample_ub" not in d
        assert "sample_u" not in d

    def _exec_to_spherical(self, x, y, z):
        in_out = sc.Dataset(
            {
                'x': sc.scalar(x, unit=sc.units.m),
                'y': sc.scalar(y, unit=sc.units.m),
                'z': sc.scalar(z, unit=sc.units.m),
            }
        )
        point = sc.spatial.as_vectors(
            in_out['x'].data, in_out['y'].data, in_out['z'].data
        )
        scn.mantid._to_spherical(point, in_out)
        return in_out

    def test_spherical_conversion(self):
        x = 1.0
        y = 1.0
        z = 0.0
        spherical = self._exec_to_spherical(x, y, z)
        assert spherical['r'].value == np.sqrt(x**2 + y**2 + z**2)
        assert spherical['t'].value == np.arccos(z / np.sqrt(x**2 + y**2 + z**2))
        # Phi now should be between 0 and pi
        assert spherical['p-delta'].value == (3.0 / 4) * np.pi
        assert spherical['p-sign'].value > 0.0
        x = -1.0
        spherical = self._exec_to_spherical(x, y, z)
        assert spherical['p-delta'].value == (1.0 / 4) * np.pi
        assert spherical['p-sign'].value > 0.0
        # Phi now should be between 0 and -pi
        y = -1.0
        spherical = self._exec_to_spherical(x, y, z)
        assert spherical['p-delta'].value == (1.0 / 4) * np.pi
        assert spherical['p-sign'].value < 0.0

    def test_detector_positions(self):
        from mantid.kernel import V3D

        eventWS = mantid.CloneWorkspace(self.base_event_ws)
        comp_info = eventWS.componentInfo()
        small_offset = V3D(0.01, 0.01, 0.01)
        comp_info.setPosition(
            comp_info.source(), comp_info.samplePosition() + small_offset
        )
        moved = scn.mantid.convert_Workspace2D_to_data_array(eventWS)
        moved_det_position = moved.coords["position"]
        unmoved = scn.mantid.convert_Workspace2D_to_data_array(eventWS)
        unmoved_det_positions = unmoved.coords["position"]
        # Moving the sample accounted for in position calculations
        # but should not yield change to final detector positions
        assert np.all(
            np.isclose(moved_det_position.values, unmoved_det_positions.values)
        )

    def test_validate_units(self):
        acceptable = ["wavelength", "Wavelength"]
        for i in acceptable:
            ret = scn.mantid.validate_dim_and_get_mantid_string(i)
            assert ret == 'Wavelength'

    def test_validate_units_throws(self):
        not_acceptable = [None, "None", "wavlength", 1, 1.0, ["wavelength"]]
        for i in not_acceptable:
            with pytest.raises(RuntimeError):
                scn.mantid.validate_dim_and_get_mantid_string(i)

    def test_WorkspaceGroup_parsed_correctly(self):
        from mantid.simpleapi import CreateSampleWorkspace, GroupWorkspaces, mtd

        CreateSampleWorkspace(OutputWorkspace="ws1")
        CreateSampleWorkspace(OutputWorkspace="ws2")
        CreateSampleWorkspace(OutputWorkspace="ws3")
        GroupWorkspaces(InputWorkspaces="ws1,ws2,ws3", OutputWorkspace="NewGroup")

        converted_group = scn.from_mantid(mtd["NewGroup"])
        converted_single = scn.from_mantid(mtd["ws1"])

        assert len(converted_group) == 3
        assert sc.identical(converted_group['ws1'], converted_single)

        mtd.clear()


@pytest.mark.skipif(not memory_is_at_least_gb(8), reason='Insufficient virtual memory')
def test_load_mcstas_data():
    wsg = mantid.LoadMcStas(
        scn.data.get_path('mcstas_sans.h5'), OutputWorkspace="test_mcstas_sans_wsg"
    )
    ws = wsg[list(wsg.getNames()).index('EventData_test_mcstas_sans_wsg')]
    dg = scn.from_mantid(ws)

    for i in range(ws.getNumberHistograms()):
        np.testing.assert_array_equal(dg['data'].coords['tof'].values, ws.readX(i))
        spec = ws.getSpectrum(i)
        da_spec = dg['data']['tof', 0]['spectrum', i]
        bin_sizes = da_spec.bins.size()
        assert spec.getNumberEvents() == bin_sizes.value

        np.testing.assert_array_equal(
            spec.getTofs(), da_spec.bins.coords['tof'].values.values
        )
        np.testing.assert_array_equal(
            spec.getPulseTimesAsNumpy(), da_spec.bins.coords['pulse_time'].value.values
        )
        np.testing.assert_array_equal(spec.getWeights(), da_spec.bins.data.value.values)


def test_to_rot_from_vectors():
    a = sc.vector(value=[1, 0, 0])
    b = sc.vector(value=[0, 1, 0])
    rot = scn.mantid._rot_from_vectors(a, b)
    assert np.allclose((rot * a).value, b.value)
    rot = scn.mantid._rot_from_vectors(b, a)
    assert np.allclose((rot * b).value, a.value)


@pytest.mark.skipif(not memory_is_at_least_gb(8), reason='Insufficient virtual memory')
@pytest.mark.parametrize(
    "param_dim",
    ['tof', 'wavelength', 'energy', 'dspacing', 'Q', 'Q^2', 'energy_transfer'],
)
def test_to_workspace_2d(param_dim):
    from mantid.simpleapi import mtd

    mtd.clear()

    data_len = 2
    expected_bins = data_len + 1
    expected_number_spectra = 10

    y = sc.Variable(
        dims=['spectrum', param_dim],
        values=np.random.rand(expected_number_spectra, data_len),
        variances=np.random.rand(expected_number_spectra, data_len),
    )

    x = sc.Variable(
        dims=['spectrum', param_dim],
        values=np.arange(
            expected_number_spectra * expected_bins, dtype=np.float64
        ).reshape((expected_number_spectra, expected_bins)),
    )
    data = sc.DataArray(data=y, coords={param_dim: x})

    ws = scn.to_mantid(data, param_dim)

    assert len(ws.readX(0)) == expected_bins
    assert ws.getNumberHistograms() == expected_number_spectra
    # check that no workspaces have been leaked in the ADS
    assert len(mtd) == 0, f"Workspaces present: {mtd.getObjectNames()}"

    for i in range(expected_number_spectra):
        np.testing.assert_array_equal(ws.readX(i), x['spectrum', i].values)
        np.testing.assert_array_equal(ws.readY(i), y['spectrum', i].values)
        np.testing.assert_array_equal(ws.readE(i), np.sqrt(y['spectrum', i].variances))


def test_to_workspace_2d_handles_single_spectra():
    from mantid.simpleapi import mtd

    mtd.clear()

    expected_x = [0.0, 1.0, 2.0]
    expected_y = [10.0, 20.0, 30.0]
    expected_e = [4.0, 4.0, 4.0]

    x = sc.Variable(dims=['tof'], values=expected_x)
    y = sc.Variable(dims=['tof'], values=expected_y, variances=expected_e)
    data = sc.DataArray(data=y, coords={'tof': x})

    ws = scn.to_mantid(data, "tof")

    assert ws.getNumberHistograms() == 1

    assert np.equal(ws.readX(0), expected_x).all()
    assert np.equal(ws.readY(0), expected_y).all()
    assert np.equal(ws.readE(0), np.sqrt(expected_e)).all()


def test_to_workspace_2d_handles_single_x_array():
    from mantid.simpleapi import mtd

    mtd.clear()

    expected_x = [0.0, 1.0, 2.0]
    expected_y = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
    expected_e = [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]

    x = sc.Variable(dims=['tof'], values=expected_x)
    y = sc.Variable(
        dims=['spectrum', 'tof'],
        values=np.array(expected_y),
        variances=np.array(expected_e),
    )
    data = sc.DataArray(data=y, coords={'tof': x})

    ws = scn.to_mantid(data, "tof")

    assert ws.getNumberHistograms() == 2
    assert np.equal(ws.readX(0), expected_x).all()
    assert np.equal(ws.readX(1), expected_x).all()

    for i, (y_vals, e_vals) in enumerate(zip(expected_y, expected_e, strict=True)):
        assert np.equal(ws.readY(i), y_vals).all()
        assert np.equal(ws.readE(i), np.sqrt(e_vals)).all()


def test_attrs_with_dims():
    import mantid.simpleapi as sapi
    from mantid.kernel import FloatArrayProperty

    dataX = [1, 2, 3]
    dataY = [1, 2, 3]
    ws = sapi.CreateWorkspace(DataX=dataX, DataY=dataY, NSpec=1, UnitX="Wavelength")
    # Time series property
    sapi.AddSampleLog(ws, 'attr0', LogText='1', LogType='Number Series')
    # Single value property
    sapi.AddSampleLog(ws, 'attr1', LogText='1', LogType='Number')
    # Array property (not time series)
    p = FloatArrayProperty('attr2', np.arange(10))
    run = ws.mutableRun()
    run.addProperty('attr2', p, replace=True)

    dg = scn.from_mantid(ws)
    assert dg['attr0'].dtype == sc.DType.int32
    assert 'time' in dg['attr0'].coords
    assert dg['attr1'].dtype == sc.DType.int64
    assert dg['attr2'].dtype == sc.DType.float64
    assert dg['attr2'].shape == (10,)


def test_time_series_log_extraction():
    ws = mantid.CreateWorkspace(DataX=[0, 1], DataY=[1])
    times = [
        np.datetime64(t)
        for t in ['2021-01-01T00:00:00', '2021-01-01T00:30:00', '2021-01-01T00:50:00']
    ]
    for i, t in enumerate(times):
        mantid.AddTimeSeriesLog(ws, Name='time_log', Time=str(t), Value=float(i))
    dg = scn.from_mantid(ws)
    assert dg['time_log'].coords['time'].dtype == sc.DType.datetime64
    # check times
    assert sc.identical(
        sc.Variable(dims=['time'], values=np.array(times).astype('datetime64[ns]')),
        dg['time_log'].coords['time'],
    )
    # check values
    assert sc.identical(
        sc.Variable(dims=['time'], values=np.arange(3.0)),
        dg['time_log'].data,
    )
    mantid.DeleteWorkspace(ws)


def test_from_mask_workspace():
    from os import path

    from mantid.simpleapi import LoadMask

    dir_path = path.dirname(path.realpath(__file__))
    mask = LoadMask('HYS', path.join(dir_path, 'HYS_mask.xml'))
    da = scn.from_mantid(mask)['data']
    assert da.unit is None
    assert da.data.dtype == sc.DType.bool
    assert da.dims == ('spectrum',)
    assert da.variances is None


def _all_indirect(blacklist):
    from mantid.simpleapi import config

    # Any indirect instrument considered
    for f in config.getFacilities():
        for i in f.instruments():
            if i.name() not in blacklist and [
                t for t in i.techniques() if 'Indirect' in t
            ]:
                yield i.name()


def _load_indirect_instrument(instr, parameters):
    from mantid.simpleapi import (
        AddSampleLog,
        LoadEmptyInstrument,
        LoadParameterFile,
        config,
    )

    # Create a workspace from an indirect instrument
    out = LoadEmptyInstrument(InstrumentName=instr)
    if instr in parameters:
        LoadParameterFile(
            out,
            Filename=os.path.join(config.getInstrumentDirectory(), parameters[instr]),
        )
    if not out.run().hasProperty('EMode'):
        # EMode would usually get attached via data loading
        # We skip that so have to apply manually
        AddSampleLog(out, LogName='EMode', LogText='Indirect', LogType='String')
    return out


def test_extract_energy_final():
    # Efinal is often stored in a non-default parameter file
    parameters = {
        'IN16B': 'IN16B_silicon_311_Parameters.xml',
        'IRIS': 'IRIS_mica_002_Parameters.xml',
        'OSIRIS': 'OSIRIS_graphite_002_Parameters.xml',
        'BASIS': 'BASIS_silicon_311_Parameters.xml',
    }
    unsupported = ['ZEEMANS', 'MARS', 'IN10', 'IN13', 'IN16', 'VISION', 'VESUVIO']
    for instr in _all_indirect(blacklist=unsupported):
        out = _load_indirect_instrument(instr, parameters)
        da = scn.from_mantid(out)['data']
        efs = da.coords["final_energy"]
        assert not sc.all(sc.isnan(efs)).value
        assert efs.unit == sc.Unit("meV")


def test_extract_energy_final_when_not_present():
    from mantid.kernel import DeltaEModeType
    from mantid.simpleapi import CreateSampleWorkspace

    ws = CreateSampleWorkspace(StoreInADS=False)
    assert ws.getEMode() == DeltaEModeType.Elastic
    da = scn.from_mantid(ws)['data']
    assert "final_energy" not in da.coords


def test_extract_energy_initial():
    from mantid.simpleapi import mtd

    mtd.clear()
    dg = scn.load_with_mantid(
        scn.data.get_path("CNCS_51936_event.nxs"), mantid_args={"SpectrumMax": 1}
    )
    assert sc.identical(
        dg['data'].coords["incident_energy"], sc.scalar(value=3.0, unit=sc.Unit("meV"))
    )


def test_extract_energy_inital_when_not_present():
    from mantid.kernel import DeltaEModeType
    from mantid.simpleapi import CreateSampleWorkspace

    ws = CreateSampleWorkspace(StoreInADS=False)
    assert ws.getEMode() == DeltaEModeType.Elastic
    dg = scn.from_mantid(ws)
    assert "incident_energy" not in dg['data'].coords


def test_EventWorkspace_with_pulse_times():
    small_event_ws = mantid.CreateSampleWorkspace(
        WorkspaceType='Event', NumBanks=1, NumEvents=10
    )
    d = scn.mantid.convert_EventWorkspace_to_data_group(
        small_event_ws, load_pulse_times=True
    )
    assert d['data'].values[0].coords['pulse_time'].dtype == sc.DType.datetime64
    assert sc.identical(
        d['data'].values[0].coords['pulse_time']['event', 0],
        sc.scalar(
            value=small_event_ws.getSpectrum(0).getPulseTimes()[0].to_datetime64()
        ),
    )


@pytest.mark.filterwarnings("ignore:convert_EventWorkspace_to_data_array")
def test_EventWorkspace_with_pulse_times_array():
    small_event_ws = mantid.CreateSampleWorkspace(
        WorkspaceType='Event', NumBanks=1, NumEvents=10
    )
    d = scn.mantid.convert_EventWorkspace_to_data_array(
        small_event_ws, load_pulse_times=True
    )
    assert d.data.values[0].coords['pulse_time'].dtype == sc.DType.datetime64
    assert sc.identical(
        d.data.values[0].coords['pulse_time']['event', 0],
        sc.scalar(
            value=small_event_ws.getSpectrum(0).getPulseTimes()[0].to_datetime64()
        ),
    )


def test_duplicate_monitor_names():
    from mantid.simpleapi import LoadEmptyInstrument

    ws = LoadEmptyInstrument(
        InstrumentName='POLARIS', StoreInADS=False
    )  # Has many monitors named 'monitor'
    monitors = scn.mantid.from_mantid(ws)["monitors"]
    assert monitors['monitor_1']['data'].coords['spectrum'].value == 1
    assert monitors['monitor_13']['data'].coords['spectrum'].value == 13
    assert monitors['monitor_14']['data'].coords['spectrum'].value == 14


def test_load_error_when_file_not_found_via_fuzzy_match():
    with pytest.raises(ValueError, match='fictional.nxs'):
        scn.load_with_mantid("fictional.nxs")


def make_dynamic_algorithm_without_fileproperty(alg_name):
    from mantid.api import (
        AlgorithmFactory,
        PythonAlgorithm,
        WorkspaceFactory,
        WorkspaceProperty,
    )
    from mantid.kernel import Direction

    # Loader without FileProperty
    if AlgorithmFactory.exists(alg_name):
        return

    class Alg(PythonAlgorithm):
        def PyInit(self):
            self.declareProperty("Filename", "")
            self.declareProperty(
                WorkspaceProperty(
                    name="OutputWorkspace", defaultValue="", direction=Direction.Output
                )
            )

        def PyExec(self):
            self.setProperty("OutputWorkspace", WorkspaceFactory.createTable())

    Alg.__name__ = alg_name
    AlgorithmFactory.subscribe(Alg)
    importlib.reload(sys.modules["mantid.simpleapi"])


def test_load_error_when_file_not_found_via_exact_match():
    make_dynamic_algorithm_without_fileproperty("DummyLoader")
    with pytest.raises(ValueError, match='fictional.nxs'):
        # DummyLoader has no FileProperty and forces
        # load to evaluate the path given as an absolute path
        scn.load_with_mantid("fictional.nxs", mantid_alg="DummyLoader")


def test_load_via_exact_match():
    make_dynamic_algorithm_without_fileproperty("DummyLoader")
    # scn.load_with_mantid will need to check file exists
    # DummyLoader simply returns a TableWorkspace
    with tempfile.NamedTemporaryFile() as fp:
        scn.load_with_mantid(fp.name, mantid_alg="DummyLoader")
        # Sanity check corrupt full path will fail
        with pytest.raises(ValueError, match='fictional_'):
            scn.load_with_mantid("fictional_" + fp.name, mantid_alg="DummyLoader")
