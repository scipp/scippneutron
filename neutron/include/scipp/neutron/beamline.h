// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <scipp/dataset/dataset.h>
#include <scipp/units/unit.h>

#include "scippneutron_export.h"

namespace scipp::neutron {

SCIPPNEUTRON_EXPORT variable::VariableConstView
position(const dataset::DatasetConstView &d);
SCIPPNEUTRON_EXPORT variable::VariableConstView
source_position(const dataset::DatasetConstView &d);
SCIPPNEUTRON_EXPORT variable::VariableConstView
sample_position(const dataset::DatasetConstView &d);
SCIPPNEUTRON_EXPORT variable::VariableView
position(const dataset::DatasetView &d);
SCIPPNEUTRON_EXPORT variable::VariableView
source_position(const dataset::DatasetView &d);
SCIPPNEUTRON_EXPORT variable::VariableView
sample_position(const dataset::DatasetView &d);
SCIPPNEUTRON_EXPORT variable::Variable
flight_path_length(const dataset::DatasetConstView &d);
SCIPPNEUTRON_EXPORT variable::Variable l1(const dataset::DatasetConstView &d);
SCIPPNEUTRON_EXPORT variable::Variable l2(const dataset::DatasetConstView &d);
SCIPPNEUTRON_EXPORT variable::Variable
scattering_angle(const dataset::DatasetConstView &d);
SCIPPNEUTRON_EXPORT variable::Variable
two_theta(const dataset::DatasetConstView &d);
SCIPPNEUTRON_EXPORT variable::VariableConstView
incident_energy(const dataset::DatasetConstView &d);
SCIPPNEUTRON_EXPORT variable::VariableConstView
final_energy(const dataset::DatasetConstView &d);

SCIPPNEUTRON_EXPORT variable::VariableConstView
position(const dataset::DataArrayConstView &d);
SCIPPNEUTRON_EXPORT variable::VariableConstView
source_position(const dataset::DataArrayConstView &d);
SCIPPNEUTRON_EXPORT variable::VariableConstView
sample_position(const dataset::DataArrayConstView &d);
SCIPPNEUTRON_EXPORT variable::VariableView
position(const dataset::DataArrayView &d);
SCIPPNEUTRON_EXPORT variable::VariableView
source_position(const dataset::DataArrayView &d);
SCIPPNEUTRON_EXPORT variable::VariableView
sample_position(const dataset::DataArrayView &d);
SCIPPNEUTRON_EXPORT variable::Variable
flight_path_length(const dataset::DataArrayConstView &d);
SCIPPNEUTRON_EXPORT variable::Variable l1(const dataset::DataArrayConstView &d);
SCIPPNEUTRON_EXPORT variable::Variable l2(const dataset::DataArrayConstView &d);
SCIPPNEUTRON_EXPORT variable::Variable
scattering_angle(const dataset::DataArrayConstView &d);
SCIPPNEUTRON_EXPORT variable::Variable
two_theta(const dataset::DataArrayConstView &d);
SCIPPNEUTRON_EXPORT variable::VariableConstView
incident_energy(const dataset::DataArrayConstView &d);
SCIPPNEUTRON_EXPORT variable::VariableConstView
final_energy(const dataset::DataArrayConstView &d);

} // namespace scipp::neutron
