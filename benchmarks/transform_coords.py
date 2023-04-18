from mantid.kernel import ConfigService

import scippneutron as scn

ConfigService.setLogLevel(1)


class TransformCoords:
    def setup(self):
        da = scn.load_with_mantid(scn.data.get_path('PG3_4844_event.nxs'))
        self.var_tof = da
        self.var_wavelength = scn.convert(self.var_tof, "tof", "wavelength", False)
        self.var_energy = scn.convert(self.var_tof, "tof", "energy", False)

    def time_wavelength_to_tof(self):
        scn.convert(self.var_tof, "tof", "wavelength", False)

    def time_tof_to_wavelength(self):
        scn.convert(self.var_wavelength, "wavelength", "tof", False)

    def time_tof_to_dspacing(self):
        scn.convert(self.var_tof, "tof", "dspacing", True)

    def time_tof_to_energy(self):
        scn.convert(self.var_tof, "tof", "energy", False)

    def time_energy_to_tof(self):
        scn.convert(self.var_energy, "energy", "tof", False)
