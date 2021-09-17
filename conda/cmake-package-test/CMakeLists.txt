# ~~~
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# ~~~
cmake_minimum_required(VERSION 3.16)
project(cmake-package-test)
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif(NOT DEFINED CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PKG_VERSION "undefined" CACHE STRING "Package version to require.")
find_package(Scippneutron ${PKG_VERSION} REQUIRED)

add_executable(test main.cpp)
target_link_libraries(test PUBLIC scippneutron)
