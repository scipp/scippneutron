// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include "test_macros.h"
#include <gtest/gtest.h>

#include <scipp/core/dimensions.h>
#include <scipp/dataset/bins.h>
#include <scipp/dataset/counts.h>
#include <scipp/dataset/dataset.h>
#include <scipp/dataset/histogram.h>
#include <scipp/variable/comparison.h>
#include <scipp/variable/operations.h>

#include "scipp/neutron/convert.h"

using namespace scipp;
using namespace scipp::neutron;

Dataset makeBeamline() {
  Dataset tof;
  static const auto source_pos = Eigen::Vector3d{0.0, 0.0, -10.0};
  static const auto sample_pos = Eigen::Vector3d{0.0, 0.0, 0.0};
  tof.setCoord(NeutronDim::SourcePosition,
               makeVariable<Eigen::Vector3d>(units::m, Values{source_pos}));
  tof.setCoord(NeutronDim::SamplePosition,
               makeVariable<Eigen::Vector3d>(units::m, Values{sample_pos}));

  tof.setCoord(NeutronDim::Position,
               makeVariable<Eigen::Vector3d>(
                   Dims{NeutronDim::Spectrum}, Shape{2}, units::m,
                   Values{Eigen::Vector3d{1.0, 0.0, 0.0},
                          Eigen::Vector3d{0.1, 0.0, 1.0}}));
  return tof;
}

Dataset makeTofDataset() {
  Dataset tof = makeBeamline();
  tof.setCoord(NeutronDim::Tof,
               makeVariable<double>(Dims{NeutronDim::Tof}, Shape{4}, units::us,
                                    Values{4000, 5000, 6100, 7300}));
  tof.setData("counts",
              makeVariable<double>(Dims{NeutronDim::Spectrum, NeutronDim::Tof},
                                   Shape{2, 3}, units::counts,
                                   Values{1, 2, 3, 4, 5, 6}));

  return tof;
}

Variable makeTofBucketedEvents() {
  Variable indices = makeVariable<std::pair<scipp::index, scipp::index>>(
      Dims{NeutronDim::Spectrum}, Shape{2},
      Values{std::pair{0, 4}, std::pair{4, 7}});
  Variable tofs =
      makeVariable<double>(Dims{Dim::Event}, Shape{7}, units::us,
                           Values{1000, 3000, 2000, 4000, 5000, 6000, 3000});
  Variable weights =
      makeVariable<double>(Dims{Dim::Event}, Shape{7}, Values{}, Variances{});
  DataArray buffer = DataArray(weights, {{NeutronDim::Tof, tofs}});
  return make_bins(std::move(indices), Dim::Event, std::move(buffer));
}

Variable makeCountDensityData(const units::Unit &unit) {
  return makeVariable<double>(Dims{NeutronDim::Spectrum, NeutronDim::Tof},
                              Shape{2, 3}, units::counts / unit,
                              Values{1, 2, 3, 4, 5, 6});
}

class ConvertTest : public testing::TestWithParam<Dataset> {};

INSTANTIATE_TEST_SUITE_P(SingleEntryDataset, ConvertTest,
                         testing::Values(makeTofDataset()));

// Tests for DataArray (or its view) as input, comparing against conversion of
// Dataset.
TEST_P(ConvertTest, DataArray_from_tof) {
  Dataset tof = GetParam();
  for (const auto &dim :
       {NeutronDim::DSpacing, NeutronDim::Wavelength, NeutronDim::Energy}) {
    const auto expected =
        convert(tof, NeutronDim::Tof, dim, ConvertMode::Scatter);
    Dataset result;
    for (const auto &data : tof)
      result.setData(data.name(),
                     convert(data, NeutronDim::Tof, dim, ConvertMode::Scatter));
    for (const auto &data : result)
      EXPECT_EQ(data, expected[data.name()]);
  }
}

TEST_P(ConvertTest, DataArray_to_tof) {
  Dataset tof = GetParam();
  for (const auto &dim :
       {NeutronDim::DSpacing, NeutronDim::Wavelength, NeutronDim::Energy}) {
    const auto input = convert(tof, NeutronDim::Tof, dim, ConvertMode::Scatter);
    const auto expected =
        convert(input, dim, NeutronDim::Tof, ConvertMode::Scatter);
    Dataset result;
    for (const auto &data : input)
      result.setData(data.name(),
                     convert(data, dim, NeutronDim::Tof, ConvertMode::Scatter));
    for (const auto &data : result)
      EXPECT_EQ(data, expected[data.name()]);
  }
}

TEST_P(ConvertTest, DataArray_non_tof) {
  Dataset tof = GetParam();
  for (const auto &from :
       {NeutronDim::DSpacing, NeutronDim::Wavelength, NeutronDim::Energy}) {
    const auto input =
        convert(tof, NeutronDim::Tof, from, ConvertMode::Scatter);
    for (const auto &to :
         {NeutronDim::DSpacing, NeutronDim::Wavelength, NeutronDim::Energy}) {
      const auto expected =
          convert(tof, NeutronDim::Tof, to, ConvertMode::Scatter);
      Dataset result;
      for (const auto &data : input)
        result.setData(data.name(),
                       convert(data, from, to, ConvertMode::Scatter));
      for (const auto &data : result)
        EXPECT_TRUE(
            all(isclose(data.coords()[to], expected.coords()[to],
                        1e-9 * units::one, 0.0 * data.coords()[to].unit()))
                .value<bool>());
    }
  }
}

TEST_P(ConvertTest, convert_slice) {
  Dataset tof = GetParam();
  const auto slice = Slice{NeutronDim::Spectrum, 0};
  for (const auto &dim :
       {NeutronDim::DSpacing, NeutronDim::Wavelength, NeutronDim::Energy}) {
    auto expected =
        convert(tof["counts"], NeutronDim::Tof, dim, ConvertMode::Scatter)
            .slice(slice);
    // A side-effect of `convert` is that it turns relevant meta data into
    // coords or attrs, depending on the target unit. Slicing (without range)
    // turns coords into attrs, but applying `convert` effectively reverses
    // this, which is why we have this slightly unusual behavior here:
    if (dim != NeutronDim::DSpacing)
      expected.coords().set(NeutronDim::Position,
                            expected.attrs().extract(NeutronDim::Position));
    EXPECT_EQ(convert(tof["counts"].slice(slice), NeutronDim::Tof, dim,
                      ConvertMode::Scatter),
              expected);
    // Converting slice of item is same as item of converted slice
    EXPECT_EQ(convert(tof["counts"].slice(slice), NeutronDim::Tof, dim,
                      ConvertMode::Scatter),
              convert(tof.slice(slice), NeutronDim::Tof, dim,
                      ConvertMode::Scatter)["counts"]);
  }
}

TEST_P(ConvertTest, fail_count_density) {
  const Dataset tof = GetParam();
  for (const Dim dim :
       {NeutronDim::DSpacing, NeutronDim::Wavelength, NeutronDim::Energy}) {
    Dataset a = tof;
    Dataset b = convert(a, NeutronDim::Tof, dim, ConvertMode::Scatter);
    EXPECT_NO_THROW(convert(a, NeutronDim::Tof, dim, ConvertMode::Scatter));
    EXPECT_NO_THROW(convert(b, dim, NeutronDim::Tof, ConvertMode::Scatter));
    a.setData("", makeCountDensityData(a.coords()[NeutronDim::Tof].unit()));
    b.setData("", makeCountDensityData(b.coords()[dim].unit()));
    EXPECT_THROW(convert(a, NeutronDim::Tof, dim, ConvertMode::Scatter),
                 except::UnitError);
    EXPECT_THROW(convert(b, dim, NeutronDim::Tof, ConvertMode::Scatter),
                 except::UnitError);
  }
}

TEST_P(ConvertTest, scattering_conversions_fail_with_NoScatter_mode) {
  Dataset tof = GetParam();
  EXPECT_THROW(convert(tof, NeutronDim::Tof, NeutronDim::DSpacing,
                       ConvertMode::NoScatter),
               std::runtime_error);
  EXPECT_NO_THROW(convert(tof, NeutronDim::Tof, NeutronDim::DSpacing,
                          ConvertMode::Scatter));
  const auto wavelength = convert(tof, NeutronDim::Tof, NeutronDim::Wavelength,
                                  ConvertMode::Scatter);
  EXPECT_THROW(convert(wavelength, NeutronDim::Wavelength, NeutronDim::Q,
                       ConvertMode::NoScatter),
               std::runtime_error);
  EXPECT_NO_THROW(convert(wavelength, NeutronDim::Wavelength, NeutronDim::Q,
                          ConvertMode::Scatter));
}

TEST_P(ConvertTest, Tof_to_DSpacing) {
  Dataset tof = GetParam();

  auto dspacing =
      convert(tof, NeutronDim::Tof, NeutronDim::DSpacing, ConvertMode::Scatter);

  ASSERT_FALSE(dspacing.coords().contains(NeutronDim::Tof));
  ASSERT_TRUE(dspacing.coords().contains(NeutronDim::DSpacing));

  const auto &coord = dspacing.coords()[NeutronDim::DSpacing];

  // Spectrum 1
  // sin(2 theta) = 0.1/(L-10)
  const double L = 10.0 + sqrt(1.0 * 1.0 + 0.1 * 0.1);
  const double lambda_to_d = 1.0 / (2.0 * sin(0.5 * asin(0.1 / (L - 10.0))));

  ASSERT_TRUE(dspacing.contains("counts"));
  EXPECT_EQ(dspacing["counts"].dims(),
            Dimensions({{NeutronDim::Spectrum, 2}, {NeutronDim::DSpacing, 3}}));
  // Due to conversion, the coordinate now also depends on NeutronDim::Spectrum.
  ASSERT_EQ(coord.dims(),
            Dimensions({{NeutronDim::Spectrum, 2}, {NeutronDim::DSpacing, 4}}));
  EXPECT_EQ(coord.unit(), units::angstrom);

  const auto values = coord.values<double>();
  // Rule of thumb (https://www.psi.ch/niag/neutron-physics):
  // v [m/s] = 3956 / \lambda [ Angstrom ]
  Variable tof_in_seconds = tof.coords()[NeutronDim::Tof] * (1e-6 * units::one);
  const auto tofs = tof_in_seconds.values<double>();
  // Spectrum 0 is 11 m from source
  // 2d sin(theta) = n \lambda
  // theta = 45 deg => d = lambda / (2 * 1 / sqrt(2))
  EXPECT_NEAR(values[0], 3956.0 / (11.0 / tofs[0]) / sqrt(2.0),
              values[0] * 1e-3);
  EXPECT_NEAR(values[1], 3956.0 / (11.0 / tofs[1]) / sqrt(2.0),
              values[1] * 1e-3);
  EXPECT_NEAR(values[2], 3956.0 / (11.0 / tofs[2]) / sqrt(2.0),
              values[2] * 1e-3);
  EXPECT_NEAR(values[3], 3956.0 / (11.0 / tofs[3]) / sqrt(2.0),
              values[3] * 1e-3);
  // Spectrum 1
  EXPECT_NEAR(values[4], 3956.0 / (L / tofs[0]) * lambda_to_d,
              values[4] * 1e-3);
  EXPECT_NEAR(values[5], 3956.0 / (L / tofs[1]) * lambda_to_d,
              values[5] * 1e-3);
  EXPECT_NEAR(values[6], 3956.0 / (L / tofs[2]) * lambda_to_d,
              values[6] * 1e-3);
  EXPECT_NEAR(values[7], 3956.0 / (L / tofs[3]) * lambda_to_d,
              values[7] * 1e-3);

  const auto &data = dspacing["counts"];
  ASSERT_EQ(data.dims(),
            Dimensions({{NeutronDim::Spectrum, 2}, {NeutronDim::DSpacing, 3}}));
  EXPECT_TRUE(equals(data.values<double>(), {1, 2, 3, 4, 5, 6}));
  EXPECT_EQ(data.unit(), units::counts);
  ASSERT_EQ(dspacing["counts"].attrs()[NeutronDim::Position],
            tof.coords()[NeutronDim::Position]);

  ASSERT_FALSE(dspacing.coords().contains(NeutronDim::Position));
  ASSERT_EQ(dspacing.coords()[NeutronDim::SourcePosition],
            tof.coords()[NeutronDim::SourcePosition]);
  ASSERT_EQ(dspacing.coords()[NeutronDim::SamplePosition],
            tof.coords()[NeutronDim::SamplePosition]);
}

TEST_P(ConvertTest, DSpacing_to_Tof) {
  /* Assuming the Tof_to_DSpacing test is correct and passing we can test the
   * inverse conversion by simply comparing a round trip conversion with the
   * original data. */

  const Dataset tof_original = GetParam();
  const auto dspacing = convert(tof_original, NeutronDim::Tof,
                                NeutronDim::DSpacing, ConvertMode::Scatter);
  const auto tof = convert(dspacing, NeutronDim::DSpacing, NeutronDim::Tof,
                           ConvertMode::Scatter);

  ASSERT_TRUE(tof.contains("counts"));
  /* Broadcasting is needed as conversion introduces the dependance on
   * NeutronDim::Spectrum */
  const auto expected_tofs = broadcast(tof_original.coords()[NeutronDim::Tof],
                                       tof.coords()[NeutronDim::Tof].dims());
  EXPECT_TRUE(equals(tof.coords()[NeutronDim::Tof].values<double>(),
                     expected_tofs.values<double>(), 1e-12));

  ASSERT_EQ(tof.coords()[NeutronDim::Position],
            tof_original.coords()[NeutronDim::Position]);
  ASSERT_EQ(tof.coords()[NeutronDim::SourcePosition],
            tof_original.coords()[NeutronDim::SourcePosition]);
  ASSERT_EQ(tof.coords()[NeutronDim::SamplePosition],
            tof_original.coords()[NeutronDim::SamplePosition]);
}

TEST_P(ConvertTest, Tof_to_Wavelength) {
  Dataset tof = GetParam();

  auto wavelength = convert(tof, NeutronDim::Tof, NeutronDim::Wavelength,
                            ConvertMode::Scatter);

  ASSERT_FALSE(wavelength.coords().contains(NeutronDim::Tof));
  ASSERT_TRUE(wavelength.coords().contains(NeutronDim::Wavelength));

  const auto &coord = wavelength.coords()[NeutronDim::Wavelength];

  ASSERT_TRUE(wavelength.contains("counts"));
  EXPECT_EQ(
      wavelength["counts"].dims(),
      Dimensions({{NeutronDim::Spectrum, 2}, {NeutronDim::Wavelength, 3}}));
  // Due to conversion, the coordinate now also depends on NeutronDim::Spectrum.
  ASSERT_EQ(coord.dims(), Dimensions({{NeutronDim::Spectrum, 2},
                                      {NeutronDim::Wavelength, 4}}));
  EXPECT_EQ(coord.unit(), units::angstrom);

  const auto values = coord.values<double>();
  // Rule of thumb (https://www.psi.ch/niag/neutron-physics):
  // v [m/s] = 3956 / \lambda [ Angstrom ]
  Variable tof_in_seconds = tof.coords()[NeutronDim::Tof] * (1e-6 * units::one);
  const auto tofs = tof_in_seconds.values<double>();
  // Spectrum 0 is 11 m from source
  EXPECT_NEAR(values[0], 3956.0 / (11.0 / tofs[0]), values[0] * 1e-3);
  EXPECT_NEAR(values[1], 3956.0 / (11.0 / tofs[1]), values[1] * 1e-3);
  EXPECT_NEAR(values[2], 3956.0 / (11.0 / tofs[2]), values[2] * 1e-3);
  EXPECT_NEAR(values[3], 3956.0 / (11.0 / tofs[3]), values[3] * 1e-3);
  // Spectrum 1
  const double L = 10.0 + sqrt(1.0 * 1.0 + 0.1 * 0.1);
  EXPECT_NEAR(values[4], 3956.0 / (L / tofs[0]), values[4] * 1e-3);
  EXPECT_NEAR(values[5], 3956.0 / (L / tofs[1]), values[5] * 1e-3);
  EXPECT_NEAR(values[6], 3956.0 / (L / tofs[2]), values[6] * 1e-3);
  EXPECT_NEAR(values[7], 3956.0 / (L / tofs[3]), values[7] * 1e-3);

  ASSERT_TRUE(wavelength.contains("counts"));
  const auto &data = wavelength["counts"];
  ASSERT_EQ(data.dims(), Dimensions({{NeutronDim::Spectrum, 2},
                                     {NeutronDim::Wavelength, 3}}));
  EXPECT_TRUE(equals(data.values<double>(), {1, 2, 3, 4, 5, 6}));
  EXPECT_EQ(data.unit(), units::counts);

  for (const auto &dim : {NeutronDim::Position, NeutronDim::SourcePosition,
                          NeutronDim::SamplePosition})
    ASSERT_EQ(wavelength.coords()[dim], tof.coords()[dim]);
}

TEST_P(ConvertTest, Wavelength_to_Tof) {
  // Assuming the Tof_to_Wavelength test is correct and passing we can test the
  // inverse conversion by simply comparing a round trip conversion with the
  // original data.

  const Dataset tof_original = GetParam();
  const auto wavelength = convert(tof_original, NeutronDim::Tof,
                                  NeutronDim::Wavelength, ConvertMode::Scatter);
  const auto tof = convert(wavelength, NeutronDim::Wavelength, NeutronDim::Tof,
                           ConvertMode::Scatter);

  ASSERT_TRUE(tof.contains("counts"));
  // Broadcasting is needed as conversion introduces the dependance on
  // NeutronDim::Spectrum
  EXPECT_TRUE(all(isclose(tof.coords()[NeutronDim::Tof],
                          tof_original.coords()[NeutronDim::Tof],
                          0.0 * units::one, 1e-12 * units::us))
                  .value<bool>());

  ASSERT_EQ(tof.coords()[NeutronDim::Position],
            tof_original.coords()[NeutronDim::Position]);
  ASSERT_EQ(tof.coords()[NeutronDim::SourcePosition],
            tof_original.coords()[NeutronDim::SourcePosition]);
  ASSERT_EQ(tof.coords()[NeutronDim::SamplePosition],
            tof_original.coords()[NeutronDim::SamplePosition]);
}

TEST_P(ConvertTest, Tof_to_Energy_Elastic) {
  Dataset tof = GetParam();

  auto energy =
      convert(tof, NeutronDim::Tof, NeutronDim::Energy, ConvertMode::Scatter);

  ASSERT_FALSE(energy.coords().contains(NeutronDim::Tof));
  ASSERT_TRUE(energy.coords().contains(NeutronDim::Energy));

  const auto &coord = energy.coords()[NeutronDim::Energy];

  constexpr auto joule_to_mev = 6.241509125883257e21;
  constexpr auto neutron_mass = 1.674927471e-27;
  /* e [J] = 1/2 m(n) [kg] (l [m] / tof [s])^2 */

  // Spectrum 1
  // sin(2 theta) = 0.1/(L-10)
  const double L = 10.0 + sqrt(1.0 * 1.0 + 0.1 * 0.1);

  ASSERT_TRUE(energy.contains("counts"));
  EXPECT_EQ(energy["counts"].dims(),
            Dimensions({{NeutronDim::Spectrum, 2}, {NeutronDim::Energy, 3}}));
  // Due to conversion, the coordinate now also depends on NeutronDim::Spectrum.
  ASSERT_EQ(coord.dims(),
            Dimensions({{NeutronDim::Spectrum, 2}, {NeutronDim::Energy, 4}}));
  EXPECT_EQ(coord.unit(), units::meV);

  const auto values = coord.values<double>();
  Variable tof_in_seconds = tof.coords()[NeutronDim::Tof] * (1e-6 * units::one);
  const auto tofs = tof_in_seconds.values<double>();

  // Spectrum 0 is 11 m from source
  EXPECT_NEAR(values[0],
              joule_to_mev * 0.5 * neutron_mass * std::pow(11 / tofs[0], 2),
              values[0] * 1e-3);
  EXPECT_NEAR(values[1],
              joule_to_mev * 0.5 * neutron_mass * std::pow(11 / tofs[1], 2),
              values[1] * 1e-3);
  EXPECT_NEAR(values[2],
              joule_to_mev * 0.5 * neutron_mass * std::pow(11 / tofs[2], 2),
              values[2] * 1e-3);
  EXPECT_NEAR(values[3],
              joule_to_mev * 0.5 * neutron_mass * std::pow(11 / tofs[3], 2),
              values[3] * 1e-3);

  // Spectrum 1
  EXPECT_NEAR(values[4],
              joule_to_mev * 0.5 * neutron_mass * std::pow(L / tofs[0], 2),
              values[4] * 1e-3);
  EXPECT_NEAR(values[5],
              joule_to_mev * 0.5 * neutron_mass * std::pow(L / tofs[1], 2),
              values[5] * 1e-3);
  EXPECT_NEAR(values[6],
              joule_to_mev * 0.5 * neutron_mass * std::pow(L / tofs[2], 2),
              values[6] * 1e-3);
  EXPECT_NEAR(values[7],
              joule_to_mev * 0.5 * neutron_mass * std::pow(L / tofs[3], 2),
              values[7] * 1e-3);

  ASSERT_TRUE(energy.contains("counts"));
  const auto &data = energy["counts"];
  ASSERT_EQ(data.dims(),
            Dimensions({{NeutronDim::Spectrum, 2}, {NeutronDim::Energy, 3}}));
  EXPECT_TRUE(equals(data.values<double>(), {1, 2, 3, 4, 5, 6}));
  EXPECT_EQ(data.unit(), units::counts);

  for (const auto &dim : {NeutronDim::Position, NeutronDim::SourcePosition,
                          NeutronDim::SamplePosition})
    ASSERT_EQ(energy.coords()[dim], tof.coords()[dim]);
}

TEST_P(ConvertTest, Tof_to_Energy_Elastic_fails_if_inelastic_params_present) {
  // Note these conversion fail only because they are not implemented. It should
  // definitely be possible to support this.
  Dataset tof = GetParam();
  EXPECT_NO_THROW_DISCARD(
      convert(tof, NeutronDim::Tof, NeutronDim::Energy, ConvertMode::Scatter));
  tof.coords().set(NeutronDim::IncidentEnergy, 2.1 * units::meV);
  EXPECT_THROW_DISCARD(
      convert(tof, NeutronDim::Tof, NeutronDim::Energy, ConvertMode::Scatter),
      std::runtime_error);
  tof.coords().erase(NeutronDim::IncidentEnergy);
  EXPECT_NO_THROW_DISCARD(
      convert(tof, NeutronDim::Tof, NeutronDim::Energy, ConvertMode::Scatter));
  tof.coords().set(NeutronDim::FinalEnergy, 2.1 * units::meV);
  EXPECT_THROW_DISCARD(
      convert(tof, NeutronDim::Tof, NeutronDim::Energy, ConvertMode::Scatter),
      std::runtime_error);
}

TEST_P(ConvertTest, Energy_to_Tof_Elastic) {
  /* Assuming the Tof_to_Energy_Elastic test is correct and passing we can test
   * the inverse conversion by simply comparing a round trip conversion with
   * the original data. */

  const Dataset tof_original = GetParam();
  const auto energy = convert(tof_original, NeutronDim::Tof, NeutronDim::Energy,
                              ConvertMode::Scatter);
  const auto tof = convert(energy, NeutronDim::Energy, NeutronDim::Tof,
                           ConvertMode::Scatter);

  ASSERT_TRUE(tof.contains("counts"));
  /* Broadcasting is needed as conversion introduces the dependance on
   * NeutronDim::Spectrum */
  const auto expected = broadcast(tof_original.coords()[NeutronDim::Tof],
                                  tof.coords()[NeutronDim::Tof].dims());
  EXPECT_TRUE(equals(tof.coords()[NeutronDim::Tof].values<double>(),
                     expected.values<double>(), 1e-12));

  ASSERT_EQ(tof.coords()[NeutronDim::Position],
            tof_original.coords()[NeutronDim::Position]);
  ASSERT_EQ(tof.coords()[NeutronDim::SourcePosition],
            tof_original.coords()[NeutronDim::SourcePosition]);
  ASSERT_EQ(tof.coords()[NeutronDim::SamplePosition],
            tof_original.coords()[NeutronDim::SamplePosition]);
}

TEST_P(ConvertTest, Tof_to_EnergyTransfer) {
  Dataset tof = GetParam();
  EXPECT_THROW_DISCARD(convert(tof, NeutronDim::Tof, NeutronDim::EnergyTransfer,
                               ConvertMode::Scatter),
                       std::runtime_error);
  tof.coords().set(NeutronDim::IncidentEnergy, 35.0 * units::meV);
  const auto direct = convert(tof, NeutronDim::Tof, NeutronDim::EnergyTransfer,
                              ConvertMode::Scatter);
  auto tof_direct = convert(direct, NeutronDim::EnergyTransfer, NeutronDim::Tof,
                            ConvertMode::Scatter);
  ASSERT_TRUE(all(isclose(tof_direct.coords()[NeutronDim::Tof],
                          tof.coords()[NeutronDim::Tof], 0.0 * units::one,
                          1e-11 * units::us))
                  .value<bool>());
  tof_direct.coords().set(NeutronDim::Tof, tof.coords()[NeutronDim::Tof]);
  EXPECT_EQ(tof_direct, tof);

  tof.coords().set(NeutronDim::FinalEnergy, 35.0 * units::meV);
  EXPECT_THROW_DISCARD(convert(tof, NeutronDim::Tof, NeutronDim::EnergyTransfer,
                               ConvertMode::Scatter),
                       std::runtime_error);
  tof.coords().erase(NeutronDim::IncidentEnergy);
  const auto indirect = convert(
      tof, NeutronDim::Tof, NeutronDim::EnergyTransfer, ConvertMode::Scatter);
  auto tof_indirect = convert(indirect, NeutronDim::EnergyTransfer,
                              NeutronDim::Tof, ConvertMode::Scatter);
  ASSERT_TRUE(all(isclose(tof_indirect.coords()[NeutronDim::Tof],
                          tof.coords()[NeutronDim::Tof], 0.0 * units::one,
                          1e-12 * units::us))
                  .value<bool>());
  tof_indirect.coords().set(NeutronDim::Tof, tof.coords()[NeutronDim::Tof]);
  EXPECT_EQ(tof_indirect, tof);

  EXPECT_NE(direct.coords()[NeutronDim::EnergyTransfer],
            indirect.coords()[NeutronDim::EnergyTransfer]);
}

TEST_P(ConvertTest, convert_with_factor_type_promotion) {
  Dataset tof = GetParam();
  tof.setCoord(NeutronDim::Tof,
               makeVariable<float>(Dims{NeutronDim::Tof}, Shape{4}, units::us,
                                   Values{4000, 5000, 6100, 7300}));
  for (auto &&d :
       {NeutronDim::DSpacing, NeutronDim::Wavelength, NeutronDim::Energy}) {
    auto res = convert(tof, NeutronDim::Tof, d, ConvertMode::Scatter);
    EXPECT_EQ(res.coords()[d].dtype(), core::dtype<float>);

    res = convert(res, d, NeutronDim::Tof, ConvertMode::Scatter);
    EXPECT_EQ(res.coords()[NeutronDim::Tof].dtype(), core::dtype<float>);
  }
}

TEST(ConvertBucketsTest, events_converted) {
  Dataset tof = makeTofDataset();
  // Standard dense coord for comparison purposes. The final 0 is a dummy.
  const auto coord = makeVariable<double>(
      Dims{NeutronDim::Spectrum, NeutronDim::Tof}, Shape{2, 4}, units::us,
      Values{1000, 3000, 2000, 4000, 5000, 6000, 3000, 0});
  tof.coords().set(NeutronDim::Tof, coord);
  tof.setData("bucketed", makeTofBucketedEvents());
  for (auto &&d :
       {NeutronDim::DSpacing, NeutronDim::Wavelength, NeutronDim::Energy}) {
    auto res = convert(tof, NeutronDim::Tof, d, ConvertMode::Scatter);
    auto values = res["bucketed"].values<bucket<DataArray>>();
    Variable expected(
        res.coords()[d].slice({NeutronDim::Spectrum, 0}).slice({d, 0, 4}));
    expected.rename(d, Dim::Event);
    EXPECT_FALSE(values[0].coords().contains(NeutronDim::Tof));
    EXPECT_TRUE(values[0].coords().contains(d));
    EXPECT_EQ(values[0].coords()[d], expected);
    expected = Variable(
        res.coords()[d].slice({NeutronDim::Spectrum, 1}).slice({d, 0, 3}));
    expected.rename(d, Dim::Event);
    EXPECT_FALSE(values[1].coords().contains(NeutronDim::Tof));
    EXPECT_TRUE(values[1].coords().contains(d));
    EXPECT_EQ(values[1].coords()[d], expected);
  }
}
