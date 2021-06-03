// SPDX-License-Identifier: BSD-3-Clause
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
  m.def(
      "position", [](const T &self) { return position(self.meta()); }, R"(
    Extract the detector pixel positions from a data array or a dataset.

    :return: A variable containing the detector pixel positions.
    :rtype: Variable)");

  m.def(
      "source_position",
      [](const T &self) { return source_position(self.meta()); },
      R"(
    Extract the neutron source position from a data array or a dataset.

    :return: A scalar variable containing the source position.
    :rtype: Variable)");

  m.def(
      "sample_position",
      [](const T &self) { return sample_position(self.meta()); },
      R"(
    Extract the sample position from a data array or a dataset.

    :return: A scalar variable containing the sample position.
    :rtype: Variable)");

  m.def(
      "Ltotal",
      [](const T &self, const bool scatter) {
        return Ltotal(self.meta(),
                      scatter ? ConvertMode::Scatter : ConvertMode::NoScatter);
      },
      py::arg("data"), py::arg("scatter"),
      R"(
    Compute the length of the total flight path from a data array or a dataset.

    If `scatter=True` this is defined as the sum of `L1` and `L2`, otherwise the distance between `source_position` and `position`.

    :return: A scalar variable containing the total length of the flight path.
    :rtype: Variable)");

  m.def(
      "incident_beam", [](const T &self) { return incident_beam(self.meta()); },
      R"(
    Compute the indicent beam vector, the direction and length of the primary flight path from a data array or a dataset.

    :return: A scalar variable containing the incident beam vector.
    :rtype: Variable)");

  m.def(
      "scattered_beam",
      [](const T &self) { return scattered_beam(self.meta()); }, R"(
    Compute the scattered beam, the directions and lengths of the secondary flight paths from a data array or a dataset.

    :return: A variable containing the scattered beam vectors for all detector pixels.
    :rtype: Variable)");

  m.def(
      "L1", [](const T &self) { return L1(self.meta()); }, R"(
    Compute L1, the length of the primary flight path (distance between neutron source and sample) from a data array or a dataset.

    :return: A scalar variable containing L1.
    :rtype: Variable)");

  m.def(
      "L2", [](const T &self) { return L2(self.meta()); }, R"(
    Compute L2, the length of the secondary flight paths (distances between sample and detector pixels) from a data array or a dataset.

    :return: A variable containing L2 for all detector pixels.
    :rtype: Variable)");

  m.def(
      "two_theta", [](const T &self) { return two_theta(self.meta()); }, R"(
    Compute :math:`2\theta`, twice the scattering angle in Bragg's law, from a data array or a dataset.

    :return: A variable containing :math:`2\theta` for all detector pixels.
    :rtype: Variable)");
}

template <class T> void bind_convert(py::module &m) {
  const char *doc = R"(
    Convert dimension (unit) into another.

    :param data: Input data
    :param origin: Dimension to convert from
    :param target: Dimension to convert into
    :param scatter: If `True` conversion with scattering from `sample_position` is performed, else non-scattering conversion is attempted.
    :param out: Optional output container
    :return: New data array or dataset with converted dimension (dimension labels, coordinate values, and units)
    :rtype: DataArray or Dataset)";
  m.def(
      "convert",
      [](T data, const Dim origin, const Dim target, const bool scatter) {
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
