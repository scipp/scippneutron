from dataclasses import dataclass

import scipp as sc

from scippneutron.atoms import ScatteringParams, reference_wavelength


@dataclass
class Material:
    scattering_params: ScatteringParams
    effective_sample_number_density: sc.Variable
    '''Density of the sample in number of formulas per cubic angstrom'''

    def attenuation_coefficient(self, wavelength: sc.Variable) -> sc.Variable:
        '''Computes marginal attenuation per distance for
        the given neutron wavelength.'''
        return self.effective_sample_number_density * (
            self.scattering_params.total_scattering_cross_section
            + (
                self.scattering_params.absorption_cross_section
                * (wavelength / reference_wavelength().to(unit=wavelength.unit))
            ).to(unit=self.scattering_params.total_scattering_cross_section.unit)
        )
