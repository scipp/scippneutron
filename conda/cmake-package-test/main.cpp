#include <scipp/dataset/data_array.h>
#include <scipp/neutron/beamline.h>

int main() {
  scipp::DataArray da(
      scipp::variable::makeVariable<double>(scipp::variable::Values{1.0}));
  da + da;
  try {
    scipp::neutron::L1(da.coords());
  } catch (scipp::except::NotFoundError &) {
  }
}
