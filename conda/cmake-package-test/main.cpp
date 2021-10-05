#include <scipp/dataset/data_array.h>
#include <scipp/neutron/convert.h>

int main() {
  scipp::DataArray da(
      scipp::variable::makeVariable<double>(scipp::variable::Values{1.0}));
  da + da;
  try {
    scipp::neutron::convert(da, scipp::neutron::NeutronDim::Tof,
                            scipp::neutron::NeutronDim::Wavelength,
                            scipp::neutron::ConvertMode::Scatter);
  } catch (scipp::except::NotFoundError &) {
  }
}
