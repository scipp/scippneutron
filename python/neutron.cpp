// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "scipp/neutron/beamline.h"
#include "scipp/neutron/convert.h"

#include "pybind11.h"

using namespace scipp;
using namespace scipp::neutron;

namespace py = pybind11;

template <class T> void bind_beamline(py::module &m) {
  using ConstView = const typename T::const_view_type &;
  m.def(
      "position", [](ConstView self) { return position(self.meta()); }, R"(
    Extract the detector pixel positions from a data array or a dataset.

    :return: A variable containing the detector pixel positions.
    :rtype: Variable)");

  m.def(
      "source_position",
      [](ConstView self) { return source_position(self.meta()); }, R"(
    Extract the neutron source position from a data array or a dataset.

    :return: A scalar variable containing the source position.
    :rtype: Variable)");

  m.def(
      "sample_position",
      [](ConstView self) { return sample_position(self.meta()); }, R"(
    Extract the sample position from a data array or a dataset.

    :return: A scalar variable containing the sample position.
    :rtype: Variable)");

  m.def(
      "flight_path_length",
      [](ConstView self, const bool scatter) {
        return flight_path_length(self.meta(), scatter
                                                   ? ConvertMode::Scatter
                                                   : ConvertMode::NoScatter);
      },
      py::arg("data"), py::arg("scatter"),
      R"(
    Compute the length of the total flight path from a data array or a dataset.

    If `scatter=True` this is the sum of `l1` and `l2`, otherwise the distance from the source.

    :return: A scalar variable containing the total length of the flight path.
    :rtype: Variable)");

  m.def(
      "l1", [](ConstView self) { return l1(self.meta()); }, R"(
    Compute L1, the length of the primary flight path (distance between neutron source and sample) from a data array or a dataset.

    :return: A scalar variable containing L1.
    :rtype: Variable)");

  m.def(
      "l2", [](ConstView self) { return l2(self.meta()); }, R"(
    Compute L2, the length of the secondary flight paths (distances between sample and detector pixels) from a data array or a dataset.

    :return: A variable containing L2 for all detector pixels.
    :rtype: Variable)");

  m.def(
      "scattering_angle",
      [](ConstView self) { return scattering_angle(self.meta()); }, R"(
    Compute :math:`\theta`, the scattering angle in Bragg's law, from a data array or a dataset.

    :return: A variable containing :math:`\theta` for all detector pixels.
    :rtype: Variable)");

  m.def(
      "two_theta", [](ConstView self) { return two_theta(self.meta()); }, R"(
    Compute :math:`2\theta`, twice the scattering angle in Bragg's law, from a data array or a dataset.

    :return: A variable containing :math:`2\theta` for all detector pixels.
    :rtype: Variable)");
}

template <class T> void bind_convert(py::module &m) {
  using ConstView = const typename T::const_view_type &;
  const char *doc = R"(
    Convert dimension (unit) into another.

    :param data: Input data with time-of-flight dimension (Dim.Tof)
    :param origin: Dimension to convert from
    :param target: Dimension to convert into
    :param out: Optional output container
    :return: New data array or dataset with converted dimension (dimension labels, coordinate values, and units)
    :rtype: DataArray or Dataset)";
  m.def(
      "convert",
      [](ConstView data, const Dim origin, const Dim target,
         const bool scatter) {
        return py::cast(
            convert(data, origin, target,
                    scatter ? ConvertMode::Scatter : ConvertMode::NoScatter));
      },
      py::arg("data"), py::arg("origin"), py::arg("target"), py::arg("scatter"),
      py::call_guard<py::gil_scoped_release>(), doc);
  m.def(
      "convert",
      [](py::object &obj, const Dim origin, const Dim target, T &out,
         const bool scatter) {
        auto &data = obj.cast<T &>();
        if (&data != &out)
          throw std::runtime_error("Currently only out=<input> is supported");
        data = convert(std::move(data), origin, target,
                       scatter ? ConvertMode::Scatter : ConvertMode::NoScatter);
        return obj;
      },
      py::arg("data"), py::arg("origin"), py::arg("target"), py::arg("out"),
      py::arg("scatter"), py::call_guard<py::gil_scoped_release>(), doc);
}

void init_neutron(py::module &m) {
  bind_convert<dataset::DataArray>(m);
  bind_convert<dataset::Dataset>(m);
  bind_beamline<dataset::DataArray>(m);
  bind_beamline<dataset::Dataset>(m);
}
