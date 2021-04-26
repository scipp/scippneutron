// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <scipp/dataset/dataset.h>
#include <scipp/units/dim.h>

#include <scipp/neutron/beamline.h>

#include "scippneutron_export.h"

namespace scipp::neutron {

[[nodiscard]] SCIPPNEUTRON_EXPORT dataset::DataArray
convert(dataset::DataArray d, const Dim from, const Dim to,
        const ConvertMode scatter);
[[nodiscard]] SCIPPNEUTRON_EXPORT dataset::DataArray
convert(const dataset::DataArrayConstView &d, const Dim from, const Dim to,
        const ConvertMode scatter);
[[nodiscard]] SCIPPNEUTRON_EXPORT dataset::Dataset
convert(dataset::Dataset d, const Dim from, const Dim to,
        const ConvertMode scatter);
[[nodiscard]] SCIPPNEUTRON_EXPORT dataset::Dataset
convert(const dataset::DatasetConstView &d, const Dim from, const Dim to,
        const ConvertMode scatter);

} // namespace scipp::neutron
