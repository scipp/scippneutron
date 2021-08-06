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

namespace {
template <class T>
[[nodiscard]] constexpr auto
measurement_cast(const llnl::units::precise_measurement &m) {
  if constexpr (std::is_same_v<std::decay_t<T>, scipp::units::Unit>) {
    return scipp::units::Unit(m.units());
  } else {
    return m.value();
  }
}
} // namespace

namespace scipp::neutron::conversions {
Variable wavelength_from_tof(const Variable &tof, const Variable &Ltotal) {
  static constexpr auto kernel = overloaded{
      core::element::arg_list<double>, [](const auto &l, const auto &t) {
        static constexpr auto c =
            measurement_cast<decltype(t)>(tof_to_wavelength_physical_constants);
        return c / l * t;
      }};
  return transform(Ltotal, tof, kernel, "wavelength_from_tof");
}

Variable energy_from_tof(const Variable &tof, const Variable &Ltotal) {
  static constexpr auto kernel = overloaded{
      core::element::arg_list<double>, [](const auto &l, const auto &t) {
        static constexpr auto c =
            measurement_cast<decltype(t)>(tof_to_energy_physical_constants);
        return c * l * l / t / t;
      }};
  return transform(Ltotal, tof, kernel, "energy_from_tof");
}

namespace {
Variable inelastic_t0(const Variable &L12, const Variable &Eif) {
  static constexpr auto kernel = overloaded{
      core::element::arg_list<double>, [](const auto &l, const auto &e) {
        // TODO correct constant? Used like this in the old conversions.
        static constexpr auto c =
            measurement_cast<decltype(l)>(tof_to_energy_physical_constants);
        using std::sqrt;
        return sqrt(l * l * c / e);
      }};
  return transform(L12, Eif, kernel, "inelastic_t0");
}
} // namespace

Variable energy_transfer_direct_from_tof(const Variable &tof,
                                         const Variable &L1, const Variable &L2,
                                         const Variable &incident_energy) {
  static constexpr auto kernel =
      overloaded{core::element::arg_list<double>,
                 [](const auto Ei, const auto l, const auto &t, const auto t0) {
                   static constexpr auto c = measurement_cast<decltype(l)>(
                       tof_to_energy_physical_constants);
                   const auto tt0 = t - t0;
                   return Ei - c * l * l / tt0 / tt0;
                 }};
  return transform(incident_energy, L2, tof, inelastic_t0(L1, incident_energy),
                   kernel, "energy_transfer_direct_from_tof");
}

Variable dspacing_from_tof(const Variable &tof, const Variable &Ltotal,
                           const Variable &two_theta) {
  return Variable{};
}

} // namespace scipp::neutron::conversions
