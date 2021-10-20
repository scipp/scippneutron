import scipp as sc
import scippneutron as scn


class TransformCoords:
    def setup(self):
        da = sc.io.open_hdf5(scn.data.get_path('powder-event.h5'))
        self.var_tof = da
        self.var_wavelength = scn.convert(self.var_tof, "tof", "wavelength", False)
        self.var_energy = scn.convert(self.var_tof, "tof", "energy", False)

    def time_wavelength_to_tof(self):
        scn.convert(self.var_tof, "tof", "wavelength", False)

    def time_tof_to_wavelength(self):
        scn.convert(self.var_wavelength, "wavelength", "tof", False)

    def time_tof_to_energy(self):
        scn.convert(self.var_tof, "tof", "energy", False)

    def time_energy_to_tof(self):
        scn.convert(self.var_energy, "energy", "tof", False)
