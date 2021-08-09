// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Jan-Lukas Wynen

#include "scipp/neutron/conversions.h"

#include <scipp/common/overloaded.h>
#include <scipp/core/element/arg_list.h>
#include <scipp/core/value_and_variance.h>
#include <scipp/variable/to_unit.h>
#include <scipp/variable/transform.h>

#include "scipp/neutron/constants.h"

using namespace scipp::variable;
using namespace scipp::neutron::constants;

namespace {
constexpr inline auto c_two = 2.0 * llnl::units::one;
constexpr inline auto c_4pi = 4.0 * scipp::pi<double> * llnl::units::one;

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

Variable Q_from_wavelength(const Variable &wavelength,
                           const Variable &two_theta) {
  static constexpr auto kernel =
      overloaded{core::element::arg_list<double>,
                 core::transform_flags::expect_no_variance_arg<1>,
                 [](const auto &lambda, const auto &angle) {
                   static constexpr auto four_pi =
                       measurement_cast<decltype(lambda)>(c_4pi);
                   static constexpr auto two =
                       measurement_cast<decltype(lambda)>(c_two);
                   using std::sin;
                   return four_pi * sin(angle / two) / lambda;
                 }};
  return transform(wavelength, two_theta, kernel, "Q_from_wavelength");
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
                 [](const auto Ei, const auto l, const auto t, const auto t0) {
                   static constexpr auto c = measurement_cast<decltype(l)>(
                       tof_to_energy_physical_constants);
                   const auto tt0 = t - t0;
                   return Ei - c * l * l / tt0 / tt0;
                 }};
  return transform(incident_energy, L2, tof, inelastic_t0(L1, incident_energy),
                   kernel, "energy_transfer_direct_from_tof");
}

Variable energy_transfer_indirect_from_tof(const Variable &tof,
                                           const Variable &L1,
                                           const Variable &L2,
                                           const Variable &final_energy) {
  static constexpr auto kernel =
      overloaded{core::element::arg_list<double>,
                 [](const auto Ef, const auto l, const auto t, const auto t0) {
                   static constexpr auto c = measurement_cast<decltype(l)>(
                       tof_to_energy_physical_constants);
                   const auto tt0 = t - t0;
                   return c * l * l / tt0 / tt0 - Ef;
                 }};
  return transform(final_energy, L1, tof, inelastic_t0(L2, final_energy),
                   kernel, "energy_transfer_indirect_from_tof");
}

Variable dspacing_from_tof(const Variable &tof, const Variable &Ltotal,
                           const Variable &two_theta) {
  static constexpr auto kernel = overloaded{
      core::element::arg_list<double>,
      core::transform_flags::expect_no_variance_arg<2>,
      [](const auto t, const auto l, const auto angle) {
        static constexpr auto c =
            measurement_cast<decltype(l)>(tof_to_dspacing_physical_constants);
        static constexpr auto two = measurement_cast<decltype(l)>(c_two);
        using std::sin;
        return t / (c * l * sin(angle / two));
      }};
  return transform(
      tof, Ltotal,
      to_unit(two_theta, scipp::units::rad, scipp::CopyPolicy::TryAvoid),
      kernel, "dspacing_from_tof");
}

} // namespace scipp::neutron::conversions
