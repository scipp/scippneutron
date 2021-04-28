// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock

#include <scipp/dataset/dataset.h>
#include <scipp/variable/operations.h>
#include <scipp/variable/transform.h>

#include "scipp/neutron/beamline.h"
#include "scipp/neutron/logging.h"

using namespace scipp;

namespace scipp::neutron {

namespace NeutronDim {
Dim DSpacing = Dim("dspacing");
Dim Energy = Dim("energy");
Dim EnergyTransfer = Dim("energy_transfer");
Dim FinalEnergy = Dim("final_energy");
Dim IncidentBeam = Dim("incident_beam");
Dim IncidentEnergy = Dim("incident_energy");
Dim L1 = Dim("L1");
Dim L2 = Dim("L2");
Dim Ltotal = Dim("Ltotal");
Dim Position = Dim("position");
Dim Q = Dim("Q");
Dim Qx = Dim("Qx");
Dim Qy = Dim("Qy");
Dim Qz = Dim("Qz");
Dim SamplePosition = Dim("sample_position");
Dim ScatteredBeam = Dim("scattered_beam");
Dim SourcePosition = Dim("source_position");
Dim Spectrum = Dim("spectrum");
Dim Tof = Dim("tof");
Dim TwoTheta = Dim("two_theta");
Dim Wavelength = Dim("wavelength");
} // namespace NeutronDim

namespace {
void log_not_found(const Dim dim, const Dim a) {
  logging::info() << dim << " coord or attr not found, trying to compute from "
                  << a << '\n';
}

void log_not_found(const Dim dim, const Dim a, const Dim b) {
  logging::info() << dim << " coord or attr not found, trying to compute from "
                  << a << " and " << b << '\n';
}

bool find_param(const dataset::Coords &meta, const Dim dim) {
  if (meta.contains(dim)) {
    logging::info() << dim << " coord or attr found, using directly\n";
    return true;
  }
  return false;
}

auto get_param(const dataset::Coords &meta, const Dim dim) {
  if (find_param(meta, dim))
    return meta[dim];
  throw except::NotFoundError(
      to_string(dim) + " coord or attr not found and no fallback available.");
}
} // namespace

Variable position(const dataset::Coords &meta) {
  return get_param(meta, NeutronDim::Position);
}

Variable source_position(const dataset::Coords &meta) {
  return get_param(meta, NeutronDim::SourcePosition);
}

Variable sample_position(const dataset::Coords &meta) {
  return get_param(meta, NeutronDim::SamplePosition);
}

Variable Ltotal(const dataset::Coords &meta, const ConvertMode scatter) {
  // TODO Avoid copies here and below if scipp buffer ownership model is changed
  if (find_param(meta, NeutronDim::Ltotal)) {
    return copy(meta[NeutronDim::Ltotal]);
  }
  // If there is not scattering this returns the straight distance from the
  // source, as required, e.g., for monitors or imaging.
  if (scatter == ConvertMode::Scatter) {
    log_not_found(NeutronDim::Ltotal, NeutronDim::L1, NeutronDim::L2);
    return L1(meta) + L2(meta);
  } else {
    log_not_found(NeutronDim::Ltotal, NeutronDim::SourcePosition,
                  NeutronDim::Position);
    return norm(position(meta) - source_position(meta));
  }
}

Variable L1(const dataset::Coords &meta) {
  if (find_param(meta, NeutronDim::L1))
    return copy(meta[NeutronDim::L1]);
  log_not_found(NeutronDim::L1, NeutronDim::IncidentBeam);
  return norm(incident_beam(meta));
}

Variable L2(const dataset::Coords &meta) {
  if (find_param(meta, NeutronDim::L2))
    return copy(meta[NeutronDim::L2]);
  log_not_found(NeutronDim::L2, NeutronDim::ScatteredBeam);
  if (find_param(meta, NeutronDim::ScatteredBeam))
    return norm(meta[NeutronDim::ScatteredBeam]);
  // Use transform to avoid temporaries. For certain unit conversions this can
  // cause a speedup >50%. Short version would be:
  //   return norm(position(meta) - sample_position(meta));
  log_not_found(NeutronDim::ScatteredBeam, NeutronDim::SamplePosition,
                NeutronDim::Position);
  return variable::transform<core::pair_self_t<Eigen::Vector3d>>(
      position(meta), sample_position(meta),
      overloaded{
          [](const auto &x, const auto &y) { return (x - y).norm(); },
          [](const units::Unit &x, const units::Unit &y) { return x - y; }});
}

Variable scattering_angle(const dataset::Coords &meta) {
  return 0.5 * units::one * two_theta(meta);
}

Variable incident_beam(const dataset::Coords &meta) {
  if (find_param(meta, NeutronDim::IncidentBeam))
    return copy(meta[NeutronDim::IncidentBeam]);
  log_not_found(NeutronDim::IncidentBeam, NeutronDim::SourcePosition,
                NeutronDim::SamplePosition);
  return sample_position(meta) - source_position(meta);
}

Variable scattered_beam(const dataset::Coords &meta) {
  if (find_param(meta, NeutronDim::ScatteredBeam))
    return copy(meta[NeutronDim::ScatteredBeam]);
  log_not_found(NeutronDim::ScatteredBeam, NeutronDim::SamplePosition,
                NeutronDim::Position);
  return position(meta) - sample_position(meta);
}

namespace {
auto normalize(Variable &&var) {
  const auto length = norm(var);
  var /= length;
  return std::move(var);
}
} // namespace

Variable cos_two_theta(const dataset::Coords &meta) {
  if (find_param(meta, NeutronDim::TwoTheta))
    return cos(meta[NeutronDim::TwoTheta]);
  log_not_found(NeutronDim::TwoTheta, NeutronDim::IncidentBeam,
                NeutronDim::ScatteredBeam);
  return dot(normalize(incident_beam(meta)), normalize(scattered_beam(meta)));
}

Variable two_theta(const dataset::Coords &meta) {
  if (find_param(meta, NeutronDim::TwoTheta))
    return copy(meta[NeutronDim::TwoTheta]);
  return acos(cos_two_theta(meta));
}

Variable incident_energy(const dataset::Coords &meta) {
  return meta.contains(NeutronDim::IncidentEnergy)
             ? meta[NeutronDim::IncidentEnergy]
             : Variable{};
}

Variable final_energy(const dataset::Coords &meta) {
  return meta.contains(NeutronDim::FinalEnergy) ? meta[NeutronDim::FinalEnergy]
                                                : Variable{};
}

} // namespace scipp::neutron
