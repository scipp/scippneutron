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
  tof.setData("counts",
              makeVariable<double>(Dims{NeutronDim::Spectrum, NeutronDim::Tof},
                                   Shape{2, 3}, units::counts,
                                   Values{1, 2, 3, 4, 5, 6}));
  tof.setCoord(NeutronDim::Tof,
               makeVariable<double>(Dims{NeutronDim::Tof}, Shape{4}, units::us,
                                    Values{4000, 5000, 6100, 7300}));

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
    for (const auto &data : tof) {
      result.setData(data.name(), convert(copy(data), NeutronDim::Tof, dim, ConvertMode::Scatter));
    }
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
