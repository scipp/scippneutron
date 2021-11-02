# ~~~
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# ~~~
python_add_library(_scippneutron MODULE neutron.cpp scippneutron.cpp)
target_include_directories(
  _scippneutron PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(_scippneutron LINK_PRIVATE scippneutron pybind11::headers)

# Set symbol visibility to hidden to reduce binary size, as recommended in
# pybind11 FAQ.
set_target_properties(_scippneutron PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(
  _scippneutron PROPERTIES INSTALL_RPATH
                           "${CMAKE_INSTALL_RPATH}/../${CMAKE_INSTALL_LIBDIR}"
)

pybind11_strip(_scippneutron)
pybind11_extension(_scippneutron)

install(TARGETS _scippneutron DESTINATION ${PYTHONDIR})
