// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Jan-Lukas Wynen

#include "scipp/neutron/conversions.h"

#include <scipp/common/overloaded.h>
#include <scipp/core/element/arg_list.h>
#include <scipp/variable/transform.h>

#include "scipp/neutron/constants.h"

using namespace scipp::variable;
using namespace scipp::neutron::constants;

namespace scipp::neutron::conversions {
namespace {
constexpr auto wavelength_from_tof_kernel = overloaded{
    core::element::arg_list<double>,
    [](const auto &Ltotal, const auto &tof) {
      return constants::tof_to_wavelength_physical_constants.value() / Ltotal *
             tof;
    },
    [](const units::Unit &Ltotal, const units::Unit &tof) {
      return units::Unit(
                 constants::tof_to_wavelength_physical_constants.units()) /
             Ltotal * tof;
    }};
}

Variable wavelength_from_tof(const Variable &Ltotal, const Variable &tof) {
  return transform(Ltotal, tof, wavelength_from_tof_kernel,
                   "wavelength_from_tof");
}
} // namespace scipp::neutron::conversions
