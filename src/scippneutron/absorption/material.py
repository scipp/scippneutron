from dataclasses import dataclass

import scipp as sc


@dataclass
class Material:
    # attenuation cross section at :math:`\lambda = 1.7982 \mathrm{angstrom}`.
    attenuation_cross_section: sc.Variable
    scattering_cross_section: sc.Variable
    effective_sample_number_density: sc.Variable

    def attenuation_coefficient(self, wavelength):
        '''Computes marginal attenuation per distance for
        the given neutron wavelength.'''
        return self.effective_sample_number_density * (
            self.scattering_cross_section
            + (
                self.attenuation_cross_section
                * (
                    wavelength
                    / sc.scalar(1.7982, unit='angstrom').to(unit=wavelength.unit)
                )
            ).to(unit=self.scattering_cross_section.unit)
        )
