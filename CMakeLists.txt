# ~~~
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# ~~~
cmake_minimum_required(VERSION 3.16)
execute_process(
  COMMAND git describe --tag --abbrev=0
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE SCIPPNEUTRON_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
project(
  scippneutron
  VERSION ${SCIPPNEUTRON_VERSION}
  LANGUAGES CXX
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE
        STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
        FORCE
  )
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Custom install target for docs to depend on.
add_custom_target(
  install-scippneutron COMMAND cmake --build ${CMAKE_CURRENT_BINARY_DIR}
                               --target install
)

include(docs)
add_docs_target(docs BUILDER html DEPENDS install-scippneutron)
add_docs_target(doctest BUILDER doctest DEPENDS docs)
add_docs_target(linkcheck BUILDER linkcheck DEPENDS docs)

include(GNUInstallDirs)
set(SITE_PACKAGES_DIR
    ""
    CACHE STRING
          "Python site-packages directory to use for installing Python module."
)
if("${SITE_PACKAGES_DIR}" STREQUAL "")
  set(PYTHONDIR scippneutron)
  set(ARCHIVEDIR .)
else()
  file(TO_CMAKE_PATH ${SITE_PACKAGES_DIR}/scippneutron PYTHONDIR)
  file(TO_CMAKE_PATH ${SITE_PACKAGES_DIR} ARCHIVEDIR)
endif()
set(INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})

enable_testing()

add_subdirectory(lib)

install(DIRECTORY "src/scippneutron/" DESTINATION ${PYTHONDIR})
