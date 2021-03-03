// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <scipp/dataset/dataset.h>
#include <scipp/units/unit.h>

#include "scippneutron_export.h"

namespace scipp::neutron {

SCIPPNEUTRON_EXPORT VariableConstView
position(const dataset::CoordsConstView &meta);
SCIPPNEUTRON_EXPORT VariableConstView
source_position(const dataset::CoordsConstView &meta);
SCIPPNEUTRON_EXPORT VariableConstView
sample_position(const dataset::CoordsConstView &meta);
SCIPPNEUTRON_EXPORT Variable
flight_path_length(const dataset::CoordsConstView &meta);
SCIPPNEUTRON_EXPORT Variable l1(const dataset::CoordsConstView &meta);
SCIPPNEUTRON_EXPORT Variable l2(const dataset::CoordsConstView &meta);
SCIPPNEUTRON_EXPORT Variable
scattering_angle(const dataset::CoordsConstView &meta);
SCIPPNEUTRON_EXPORT Variable two_theta(const dataset::CoordsConstView &meta);
SCIPPNEUTRON_EXPORT VariableConstView
incident_energy(const dataset::CoordsConstView &meta);
SCIPPNEUTRON_EXPORT VariableConstView
final_energy(const dataset::CoordsConstView &meta);

} // namespace scipp::neutron
