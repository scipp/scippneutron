// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include <set>
#include <tuple>

#include <scipp/core/element/arg_list.h>

#include <scipp/variable/bucket_model.h>
#include <scipp/variable/transform.h>
#include <scipp/variable/util.h>

#include <scipp/dataset/bins.h>
#include <scipp/dataset/dataset.h>
#include <scipp/dataset/dataset_util.h>

#include "scipp/neutron/constants.h"
#include "scipp/neutron/conversions.h"
#include "scipp/neutron/convert.h"

using namespace scipp::variable;
using namespace scipp::dataset;

namespace scipp::neutron {

template <class T, class Op, class... Args>
T convert_generic(T &&d, const Dim from, const Dim to, Op op,
                  const Args &... args) {
  using core::element::arg_list;
  const auto op_ = overloaded{
      arg_list<double,
               std::tuple<float, std::conditional_t<true, double, Args>...>>,
      op};
  const auto items = iter(d);
  // 1. Transform coordinate
  if (d.coords().contains(from)) {
    const auto coord = d.coords()[from];
    if (!coord.dims().contains(merge(args.dims()...)))
      d.coords().set(from,
                     broadcast(coord, merge(args.dims()..., coord.dims())));
    transform_in_place(d.coords()[from], args..., op_);
  }
  // 2. Transform coordinates in bucket variables
  for (const auto &item : iter(d)) {
    if (item.dtype() != dtype<bucket<DataArray>>)
      continue;
    const auto &[indices, dim, buffer] =
        item.data().template constituents<bucket<DataArray>>();
    if (!buffer.coords().contains(from))
      continue;
    auto buffer_coord = buffer.coords().extract(from);
    auto coord = make_non_owning_bins(indices, dim, VariableView(buffer_coord));
    transform_in_place(coord, args..., op_);
    buffer.coords().set(to, std::move(buffer_coord));
  }

  // 3. Rename dims
  d.rename(from, to);
  return std::move(d);
}

namespace {
template <class T, class Op, class Tuple, std::size_t... I>
T convert_arg_tuple_impl(T &&d, const Dim from, const Dim to, Op op, Tuple &&t,
                         std::index_sequence<I...>) {
  return convert_generic(std::forward<T>(d), from, to, op,
                         std::get<I>(std::forward<Tuple>(t))...);
}
} // namespace

template <class T, class Op, class Tuple>
T convert_arg_tuple(T &&d, const Dim from, const Dim to, Op op, Tuple &&t) {
  return convert_arg_tuple_impl(
      std::forward<T>(d), from, to, op, std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
}

template <class T>
static T convert_with_factor(T &&d, const Dim from, const Dim to,
                             const Variable &factor) {
  return convert_generic(
      std::forward<T>(d), from, to,
      [](auto &coord, const auto &c) { coord *= c; }, factor);
}

namespace {

const auto &no_scatter_params() {
  static const std::array<Dim, 2> params{NeutronDim::Position,
                                         NeutronDim::Ltotal};
  return params;
}

auto scatter_params(const Dim dim) {
  static std::set<Dim> pos_invariant{Dim::Invalid, NeutronDim::DSpacing,
                                     NeutronDim::Q};
  static std::vector<Dim> params{NeutronDim::Position,
                                 NeutronDim::IncidentBeam,
                                 NeutronDim::ScatteredBeam,
                                 NeutronDim::IncidentEnergy,
                                 NeutronDim::FinalEnergy,
                                 NeutronDim::Ltotal,
                                 NeutronDim::L1,
                                 NeutronDim::L2};
  if (dim == NeutronDim::Tof)
    return scipp::span<Dim>{};
  return pos_invariant.count(dim) ? scipp::span(params)
                                  : scipp::span(params).subspan(3);
}

template <class T>
T coords_to_attrs(T &&x, const Dim from, const Dim to,
                  const ConvertMode scatter) {
  const auto to_attr = [&](const Dim field) {
    if (!x.coords().contains(field))
      return;
    Variable coord(x.coords()[field]);
    if constexpr (std::is_same_v<std::decay_t<T>, Dataset>) {
      x.coords().erase(field);
      for (const auto &item : iter(x))
        item.attrs().set(field, coord);
    } else {
      x.coords().erase(field);
      x.attrs().set(field, coord);
    }
  };
  if (scatter == ConvertMode::Scatter) {
    for (const auto &param : scatter_params(to))
      to_attr(param);
  } else if (from == NeutronDim::Tof) {
    for (const auto &param : no_scatter_params())
      to_attr(param);
  }
  return std::move(x);
}

template <class T>
T attrs_to_coords(T &&x, const Dim to, const ConvertMode scatter) {
  const auto to_coord = [&](const Dim field) {
    auto &&range = iter(x);
    if (!range.begin()->attrs().contains(field))
      return;
    Variable attr(range.begin()->attrs()[field]);
    if constexpr (std::is_same_v<std::decay_t<T>, Dataset>) {
      for (const auto &item : range) {
        core::expect::equals(item.attrs()[field], attr);
        item.attrs().erase(field);
      }
      x.coords().set(field, attr);
    } else {
      x.attrs().erase(field);
      x.coords().set(field, attr);
    }
  };
  if (scatter == ConvertMode::Scatter) {
    // Before conversion we convert all geometry-related params into coords,
    // otherwise conversions with datasets will not work since attrs are
    // item-specific.
    for (const auto &param : scatter_params(Dim::Invalid))
      to_coord(param);
  } else if (to == NeutronDim::Tof) {
    for (const auto &param : no_scatter_params())
      to_coord(param);
  }
  return std::move(x);
}

void check_scattering(const Dim from, const Dim to, const ConvertMode scatter) {
  std::set<Dim> scattering{NeutronDim::DSpacing, NeutronDim::Q,
                           NeutronDim::EnergyTransfer};
  if ((scatter == ConvertMode::NoScatter) &&
      (scattering.count(from) || scattering.count(to)))
    throw std::runtime_error(
        "Conversion with `scatter=False` requested, but `" + to_string(from) +
        "` and/or `" + to_string(to) +
        "` is only defined for a scattering process.");
}

} // namespace

template <class T>
T convert_impl(T d, const Dim from, const Dim to, const ConvertMode scatter) {
  for (const auto &item : iter(d))
    core::expect::notCountDensity(item.unit());

  d = attrs_to_coords(std::move(d), to, scatter);
  // This will need to be cleanup up in the future, but it is unclear how to do
  // so in a future-proof way. Some sort of double-dynamic dispatch based on
  // `from` and `to` will likely be required (with conversions helpers created
  // by a dynamic factory based on `Dim`). Conceptually we are dealing with a
  // bidirectional graph, and we would like to be able to find the shortest
  // paths between any two points, without defining all-to-all connections.
  // Approaches based on, e.g., a map of conversions and constants is also
  // tricky, since in particular the conversions are generic lambdas (passable
  // to `transform`) and are not readily stored as function pointers or
  // std::function.
  if ((from == NeutronDim::Tof) && (to == NeutronDim::DSpacing))
    return convert_with_factor(std::move(d), from, to,
                               constants::tof_to_dspacing(d));
  if ((from == NeutronDim::DSpacing) && (to == NeutronDim::Tof))
    return convert_with_factor(std::move(d), from, to,
                               reciprocal(constants::tof_to_dspacing(d)));

  if ((from == NeutronDim::Tof) && (to == NeutronDim::Wavelength))
    return convert_with_factor(std::move(d), from, to,
                               constants::tof_to_wavelength(d, scatter));
  if ((from == NeutronDim::Wavelength) && (to == NeutronDim::Tof))
    return convert_with_factor(
        std::move(d), from, to,
        reciprocal(constants::tof_to_wavelength(d, scatter)));

  if ((from == NeutronDim::Tof) && (to == NeutronDim::Energy))
    return convert_generic(std::move(d), from, to, conversions::tof_to_energy,
                           constants::tof_to_energy(d, scatter));
  if ((from == NeutronDim::Energy) && (to == NeutronDim::Tof))
    return convert_generic(std::move(d), from, to, conversions::energy_to_tof,
                           constants::tof_to_energy(d, scatter));

  if ((from == NeutronDim::Tof) && (to == NeutronDim::EnergyTransfer))
    return convert_arg_tuple(std::move(d), from, to,
                             conversions::tof_to_energy_transfer,
                             constants::tof_to_energy_transfer(d));
  if ((from == NeutronDim::EnergyTransfer) && (to == NeutronDim::Tof))
    return convert_arg_tuple(std::move(d), from, to,
                             conversions::energy_transfer_to_tof,
                             constants::tof_to_energy_transfer(d));

  // lambda <-> Q conversion is symmetric
  if (((from == NeutronDim::Wavelength) && (to == NeutronDim::Q)) ||
      ((from == NeutronDim::Q) && (to == NeutronDim::Wavelength)))
    return convert_generic(std::move(d), from, to, conversions::wavelength_to_q,
                           constants::wavelength_to_q(d));

  try {
    // Could get better performance by doing a direct conversion.
    return convert_impl(
        convert_impl(std::move(d), from, NeutronDim::Tof, scatter),
        NeutronDim::Tof, to, scatter);
  } catch (const except::UnitError &) {
    throw except::UnitError("Conversion between " + to_string(from) + " and " +
                            to_string(to) +
                            " not implemented yet or not possible.");
  }
}

DataArray convert(DataArray d, const Dim from, const Dim to,
                  const ConvertMode scatter) {
  check_scattering(from, to, scatter);
  return coords_to_attrs(convert_impl(std::move(d), from, to, scatter), from,
                         to, scatter);
}

DataArray convert(const DataArrayConstView &d, const Dim from, const Dim to,
                  const ConvertMode scatter) {
  check_scattering(from, to, scatter);
  return convert(DataArray(d), from, to, scatter);
}

Dataset convert(Dataset d, const Dim from, const Dim to,
                const ConvertMode scatter) {
  check_scattering(from, to, scatter);
  return coords_to_attrs(convert_impl(std::move(d), from, to, scatter), from,
                         to, scatter);
}

Dataset convert(const DatasetConstView &d, const Dim from, const Dim to,
                const ConvertMode scatter) {
  check_scattering(from, to, scatter);
  return convert(Dataset(d), from, to, scatter);
}

} // namespace scipp::neutron
