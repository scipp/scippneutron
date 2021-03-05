// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock

#include <scipp/dataset/dataset.h>
#include <scipp/variable/operations.h>
#include <scipp/variable/transform.h>

#include "scipp/neutron/beamline.h"

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

VariableConstView position(const dataset::CoordsConstView &meta) {
  return meta[NeutronDim::Position];
}

VariableConstView source_position(const dataset::CoordsConstView &meta) {
  return meta[NeutronDim::SourcePosition];
}

VariableConstView sample_position(const dataset::CoordsConstView &meta) {
  return meta[NeutronDim::SamplePosition];
}

Variable Ltotal(const dataset::CoordsConstView &meta,
                const ConvertMode scatter) {
  // TODO Avoid copies here and below if scipp buffer ownership model is changed
  if (meta.contains(NeutronDim::Ltotal))
    return copy(meta[NeutronDim::Ltotal]);
  // If there is not scattering this returns the straight distance from the
  // source, as required, e.g., for monitors or imaging.
  if (scatter == ConvertMode::Scatter)
    return L1(meta) + L2(meta);
  else
    return norm(position(meta) - source_position(meta));
}

Variable L1(const dataset::CoordsConstView &meta) {
  if (meta.contains(NeutronDim::L1))
    return copy(meta[NeutronDim::L1]);
  return norm(incident_beam(meta));
}

Variable L2(const dataset::CoordsConstView &meta) {
  if (meta.contains(NeutronDim::L2))
    return copy(meta[NeutronDim::L2]);
  if (meta.contains(NeutronDim::ScatteredBeam))
    return norm(meta[NeutronDim::ScatteredBeam]);
  // Use transform to avoid temporaries. For certain unit conversions this can
  // cause a speedup >50%. Short version would be:
  //   return norm(position(meta) - sample_position(meta));
  return variable::transform<core::pair_self_t<Eigen::Vector3d>>(
      position(meta), sample_position(meta),
      overloaded{
          [](const auto &x, const auto &y) { return (x - y).norm(); },
          [](const units::Unit &x, const units::Unit &y) { return x - y; }});
}

Variable scattering_angle(const dataset::CoordsConstView &meta) {
  return 0.5 * units::one * two_theta(meta);
}

Variable incident_beam(const dataset::CoordsConstView &meta) {
  if (meta.contains(NeutronDim::IncidentBeam))
    return copy(meta[NeutronDim::IncidentBeam]);
  return sample_position(meta) - source_position(meta);
}

Variable scattered_beam(const dataset::CoordsConstView &meta) {
  if (meta.contains(NeutronDim::ScatteredBeam))
    return copy(meta[NeutronDim::ScatteredBeam]);
  return position(meta) - sample_position(meta);
}

namespace {
auto normalize(Variable &&var) {
  const auto length = norm(var);
  var /= length;
  return std::move(var);
}
} // namespace

Variable cos_two_theta(const dataset::CoordsConstView &meta) {
  if (meta.contains(NeutronDim::TwoTheta))
    return cos(meta[NeutronDim::TwoTheta]);
  return dot(normalize(incident_beam(meta)), normalize(scattered_beam(meta)));
}

Variable two_theta(const dataset::CoordsConstView &meta) {
  if (meta.contains(NeutronDim::TwoTheta))
    return copy(meta[NeutronDim::TwoTheta]);
  return acos(cos_two_theta(meta));
}

VariableConstView incident_energy(const dataset::CoordsConstView &meta) {
  return meta.contains(NeutronDim::IncidentEnergy)
             ? meta[NeutronDim::IncidentEnergy]
             : VariableConstView{};
}

VariableConstView final_energy(const dataset::CoordsConstView &meta) {
  return meta.contains(NeutronDim::FinalEnergy) ? meta[NeutronDim::FinalEnergy]
                                                : VariableConstView{};
}

} // namespace scipp::neutron
