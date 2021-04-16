# ~~~
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# ~~~
pybind11_add_module(_scippneutron SHARED SYSTEM neutron.cpp scippneutron.cpp)
target_include_directories(
  _scippneutron PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(_scippneutron LINK_PRIVATE scippneutron)

# Set symbol visibility to hidden to reduce binary size, as recommended in
# pybind11 FAQ.
set_target_properties(_scippneutron PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(
  _scippneutron PROPERTIES INSTALL_RPATH
                           "${CMAKE_INSTALL_RPATH}/../${CMAKE_INSTALL_LIBDIR}"
)

install(DIRECTORY "src/scippneutron/" DESTINATION ${PYTHONDIR})
install(TARGETS _scippneutron DESTINATION ${PYTHONDIR})
