// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include <iostream>

#include "scipp/neutron/logging.h"

using namespace scipp;

namespace scipp::neutron::logging {

std::ostream &info() { return std::cout << "INFO:scippneutron:"; }

} // namespace scipp::neutron::logging
