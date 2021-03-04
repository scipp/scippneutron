// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include <scipp/dataset/dataset.h>
#include <scipp/variable/operations.h>

#include "scipp/neutron/beamline.h"

using namespace scipp;
using namespace scipp::variable;
using namespace scipp::neutron;

namespace {
static const auto source_pos = Eigen::Vector3d{0.0, 0.0, -9.99};
static const auto sample_pos = Eigen::Vector3d{0.0, 0.0, 0.01};
} // namespace

Dataset makeDatasetWithBeamline() {
  Dataset beamline;
  Dataset components;
  // Source and sample
  components.setData("position", makeVariable<Eigen::Vector3d>(
                                     Dims{Dim::Row}, Shape{2}, units::m,
                                     Values{source_pos, sample_pos}));
  beamline.setCoord(Dim("source_position"), makeVariable<Eigen::Vector3d>(
                                                units::m, Values{source_pos}));
  beamline.setCoord(Dim("sample_position"), makeVariable<Eigen::Vector3d>(
                                                units::m, Values{sample_pos}));
  // TODO Need fuzzy comparison for variables to write a convenient test with
  // detectors away from the axes.
  beamline.setCoord(
      Dim("position"),
      makeVariable<Eigen::Vector3d>(Dims{Dim::Spectrum}, Shape{2}, units::m,
                                    Values{Eigen::Vector3d{1.0, 0.0, 0.01},
                                           Eigen::Vector3d{0.0, 1.0, 0.01}}));
  return beamline;
}

class BeamlineTest : public ::testing::Test {
protected:
  Dataset dataset{makeDatasetWithBeamline()};

  Variable L2_override = makeVariable<double>(Dims{Dim::Spectrum}, Shape{2},
                                              units::m, Values{1.1, 1.2});
  Variable incident_beam_override =
      incident_beam(dataset.meta()) * (1.23 * units::one);
  Variable scattered_beam_override = makeVariable<Eigen::Vector3d>(
      Dims{Dim::Spectrum}, Shape{2}, units::m,
      Values{Eigen::Vector3d{1.0, 0.1, 0.11}, Eigen::Vector3d{0.1, 1.0, 0.11}});
};

TEST_F(BeamlineTest, basics) {
  ASSERT_EQ(source_position(dataset.meta()),
            makeVariable<Eigen::Vector3d>(Dims(), Shape(), units::m,
                                          Values{source_pos}));
  ASSERT_EQ(sample_position(dataset.meta()),
            makeVariable<Eigen::Vector3d>(Dims(), Shape(), units::m,
                                          Values{sample_pos}));
}

TEST_F(BeamlineTest, l1) {
  const auto L1_override = 10.1 * units::m;
  // No overrides, computed based on positions
  ASSERT_EQ(l1(dataset.meta()), 10.0 * units::m);
  dataset.coords().set(Dim("L1"), L1_override);
  // Explicit L1 used
  ASSERT_EQ(l1(dataset.meta()), L1_override);
  dataset.coords().set(Dim("incident_beam"), incident_beam_override);
  // Explicit L1 has higher priority than incident_beam
  ASSERT_EQ(l1(dataset.meta()), L1_override);
  ASSERT_NE(l1(dataset.meta()), norm(incident_beam_override));
  dataset.coords().erase(Dim("L1"));
  // Now incident_beam is used
  ASSERT_EQ(l1(dataset.meta()), norm(incident_beam_override));
  dataset.coords().erase(Dim("incident_beam"));
  // Back to computation based on positions
  ASSERT_EQ(l1(dataset.meta()), 10.0 * units::m);
}

TEST_F(BeamlineTest, l2) {
  const auto L2_computed = makeVariable<double>(Dims{Dim::Spectrum}, Shape{2},
                                                units::m, Values{1.0, 1.0});
  // No overrides, computed based on positions
  ASSERT_EQ(l2(dataset.meta()), L2_computed);
  dataset.coords().set(Dim("L2"), L2_override);
  // Explicit L2 used
  ASSERT_EQ(l2(dataset.meta()), L2_override);
  dataset.coords().set(Dim("scattered_beam"), scattered_beam_override);
  // Explicit L2 has higher priority than scattered_beam
  ASSERT_EQ(l2(dataset.meta()), L2_override);
  ASSERT_NE(l2(dataset.meta()), norm(scattered_beam_override));
  dataset.coords().erase(Dim("L2"));
  // Now scattered_beam is used
  ASSERT_EQ(l2(dataset.meta()), norm(scattered_beam_override));
  dataset.coords().erase(Dim("scattered_beam"));
  // Back to computation based on positions
  ASSERT_EQ(l2(dataset.meta()), L2_computed);
}

TEST_F(BeamlineTest, incident_beam) {
  const auto incident_beam_computed =
      sample_position(dataset.meta()) - source_position(dataset.meta());
  ASSERT_EQ(incident_beam(dataset.meta()), incident_beam_computed);
  dataset.coords().set(Dim("incident_beam"), incident_beam_override);
  ASSERT_EQ(incident_beam(dataset.meta()), incident_beam_override);
  dataset.coords().erase(Dim("incident_beam"));
  ASSERT_EQ(incident_beam(dataset.meta()), incident_beam_computed);
}

TEST_F(BeamlineTest, scattered_beam) {
  const auto scattered_beam_computed =
      position(dataset.meta()) - sample_position(dataset.meta());
  ASSERT_EQ(scattered_beam(dataset.meta()), scattered_beam_computed);
  dataset.coords().set(Dim("scattered_beam"), scattered_beam_override);
  ASSERT_EQ(scattered_beam(dataset.meta()), scattered_beam_override);
  dataset.coords().erase(Dim("scattered_beam"));
  ASSERT_EQ(scattered_beam(dataset.meta()), scattered_beam_computed);
}

TEST_F(BeamlineTest, flight_path_length) {
  const auto L_computed = l1(dataset.meta()) + l2(dataset.meta());
  ASSERT_EQ(flight_path_length(dataset.meta(), ConvertMode::Scatter),
            L_computed);
  ASSERT_EQ(flight_path_length(dataset.meta(), ConvertMode::NoScatter),
            norm(source_position(dataset.meta()) - position(dataset.meta())));
  dataset.coords().set(Dim("L2"), L2_override);
  ASSERT_NE(flight_path_length(dataset.meta(), ConvertMode::Scatter),
            L_computed);
  ASSERT_EQ(flight_path_length(dataset.meta(), ConvertMode::Scatter),
            l1(dataset.meta()) + l2(dataset.meta()));
  // In non-scattering conversion L2 is irrelvant, so adding the coord has no
  // effect in this case
  ASSERT_EQ(flight_path_length(dataset.meta(), ConvertMode::NoScatter),
            norm(source_position(dataset.meta()) - position(dataset.meta())));
  const auto L_override = l1(dataset.meta()) + L2_override * (1.1 * units::one);
  dataset.coords().set(Dim("Ltotal"), L_override);
  // Note that now L2 is also overridden by L
  ASSERT_EQ(flight_path_length(dataset.meta(), ConvertMode::Scatter),
            L_override);
  ASSERT_EQ(flight_path_length(dataset.meta(), ConvertMode::NoScatter),
            L_override);
}

template <class T> constexpr T pi = T(3.1415926535897932385L);

TEST_F(BeamlineTest, scattering_angle) {
  const auto two_theta_computed =
      makeVariable<double>(Dims{Dim::Spectrum}, Shape{2}, units::rad,
                           Values{pi<double> / 2, pi<double> / 2});
  const auto theta_computed = 0.5 * units::one * two_theta_computed;
  const auto cos_two_theta_computed =
      makeVariable<double>(Dims{Dim::Spectrum}, Shape{2}, Values{0.0, 0.0});

  ASSERT_EQ(cos_two_theta(dataset.meta()), cos_two_theta_computed);
  ASSERT_EQ(two_theta(dataset.meta()), two_theta_computed);
  ASSERT_EQ(scattering_angle(dataset.meta()), theta_computed);

  const auto two_theta_override = makeVariable<double>(
      Dims{Dim::Spectrum}, Shape{2}, units::rad, Values{0.1, 0.2});
  // Setting `theta` or `scattering_angle` has no effect. These are slightly
  // ambiguous and are therefore not interpreted by the beamline helpers.
  dataset.coords().set(Dim("theta"), two_theta_override);
  ASSERT_EQ(cos_two_theta(dataset.meta()), cos_two_theta_computed);
  ASSERT_EQ(two_theta(dataset.meta()), two_theta_computed);
  ASSERT_EQ(scattering_angle(dataset.meta()), theta_computed);
  dataset.coords().erase(Dim("theta"));
  dataset.coords().set(Dim("scattering_angle"), two_theta_override);
  ASSERT_EQ(cos_two_theta(dataset.meta()), cos_two_theta_computed);
  ASSERT_EQ(two_theta(dataset.meta()), two_theta_computed);
  ASSERT_EQ(scattering_angle(dataset.meta()), theta_computed);
  dataset.coords().erase(Dim("scattering_angle"));

  // override via two_theta
  dataset.coords().set(Dim("two_theta"), two_theta_override);
  ASSERT_EQ(cos_two_theta(dataset.meta()), cos(two_theta_override));
  ASSERT_EQ(two_theta(dataset.meta()), two_theta_override);
  ASSERT_EQ(scattering_angle(dataset.meta()),
            0.5 * units::one * two_theta_override);

  dataset.coords().set(Dim("scattered_beam"), scattered_beam_override);
  // No change, two_theta has higher priority...
  ASSERT_EQ(cos_two_theta(dataset.meta()), cos(two_theta_override));
  ASSERT_EQ(two_theta(dataset.meta()), two_theta_override);
  ASSERT_EQ(scattering_angle(dataset.meta()),
            0.5 * units::one * two_theta_override);
  // ... erase ...
  dataset.coords().erase(Dim("two_theta"));
  // ... now scattered_beam is used
  ASSERT_NE(cos_two_theta(dataset.meta()), cos(two_theta_override));
  ASSERT_NE(two_theta(dataset.meta()), two_theta_override);
  ASSERT_NE(scattering_angle(dataset.meta()),
            0.5 * units::one * two_theta_override);
  ASSERT_EQ(
      cos_two_theta(dataset.meta()),
      dot(incident_beam(dataset.meta()) / norm(incident_beam(dataset.meta())),
          scattered_beam_override / norm(scattered_beam_override)));
}

TEST_F(BeamlineTest, no_scatter) {
  Dataset d(dataset);
  d.coords().erase(Dim("sample_position"));
  ASSERT_THROW(l1(d.meta()), except::NotFoundError);
  ASSERT_THROW(l2(d.meta()), except::NotFoundError);
  ASSERT_THROW(scattering_angle(d.meta()), except::NotFoundError);
  ASSERT_EQ(flight_path_length(d.meta(), ConvertMode::NoScatter),
            makeVariable<double>(
                Dims{Dim::Spectrum}, Shape{2}, units::m,
                Values{(Eigen::Vector3d{1.0, 0.0, 0.01} - source_pos).norm(),
                       (Eigen::Vector3d{0.0, 1.0, 0.01} - source_pos).norm()}));
}
