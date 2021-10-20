import scipp as sc
import scippneutron as scn


class Conversion:
    def setup(self):
        da = scn.data.tutorial_event_data()
        self.var_wavelength= scn.convert(da, "tof", "wavelength", False)
        self.var_tof = da
        self.var_dspacing = scn.convert(da, "tof", "dspacing", False)
        self.var_energy = scn.convert(self.var_tof, "tof", "energy", False)

    def time_wavelength_to_tof(self):
        scn.convert(self.var_tof, "tof", "wavelength", False)

    def time_tof_to_wavelength(self):
        scn.convert(self.var_wavelength, "wavelength", "tof", False)

    def time_tof_to_dspacing(self):
        scn.convert(self.var_tof, "tof", "dspacing", False)

    def time_dspacing_to_tof(self):
        scn.convert(self.var_dspacing, "dspacing", "tof", False)

    def time_tof_to_energy(self):
        scn.convert(self.var_tof, "tof", "energy", False)

    def time_energy_to_tof(self):
        scn.convert(self.var_energy, "energy", "tof", False)

    def time_tof_to_energy_transfer(self):
        scn.convert(self.var_tof, "tof", "energy_transfer", False)
