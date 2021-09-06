// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Jan-Lukas Wynen

#include "scipp/neutron/conversions.h"

#include "pybind11.h"

using namespace scipp;
using namespace scipp::neutron;

namespace py = pybind11;

void init_conversions(py::module &base_module) {
  auto m = base_module.def_submodule("conversions");
  m.def("energy_transfer_indirect_from_tof",
        conversions::energy_transfer_indirect_from_tof);
}