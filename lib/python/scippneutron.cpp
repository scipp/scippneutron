// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "pybind11.h"

namespace py = pybind11;

void init_neutron(py::module &);

PYBIND11_MODULE(_scippneutron, m) {
#ifdef SCIPPNEUTRON_VERSION
  m.attr("__version__") = py::str(SCIPPNEUTRON_VERSION);
#else
  m.attr("__version__") = py::str("unknown version");
#endif
  init_neutron(m);
}
