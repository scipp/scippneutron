# ~~~
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# ~~~
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif(NOT DEFINED CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)
if(MSVC)
  set(CLCACHE_PATH
      ""
      CACHE PATH "Path to installed clcache executable directory"
  )
  if(EXISTS ${CLCACHE_PATH})
    set(CMAKE_VS_GLOBALS "CLToolExe=clcache.exe" "CLToolPath=${CLCACHE_PATH}"
                         "TrackFileAccess=false"
    )
    message(STATUS "Will attempt to use clcache")
  else(EXISTS ${CLCACHE_PATH})
    message(STATUS "CLCACHE_PATH not specified. Not enabled.")
    unset(CMAKE_VS_GLOBALS) # Clear all vs globals
  endif(EXISTS ${CLCACHE_PATH})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /constexpr:steps2000000")
endif(MSVC)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)

execute_process(
  COMMAND conan export .
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.conan-recipes/llnl-units"
                    COMMAND_ECHO STDOUT
)

# Conan dependencies
if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/conan.cmake")
  message(
    STATUS
      "Downloading conan.cmake from https://github.com/conan-io/cmake-conan"
  )
  file(
    DOWNLOAD
    "https://raw.githubusercontent.com/conan-io/cmake-conan/v0.16.1/conan.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/conan.cmake"
    EXPECTED_HASH
      SHA256=396e16d0f5eabdc6a14afddbcfff62a54a7ee75c6da23f32f7a31bc85db23484
    TLS_VERIFY ON
  )
endif()

include(${CMAKE_CURRENT_BINARY_DIR}/conan.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

conan_cmake_autodetect(conan_settings)
conan_cmake_install(
  PATH_OR_REFERENCE ${CMAKE_CURRENT_SOURCE_DIR} SETTINGS ${conan_settings}
  BUILD outdated
)

find_package(GTest CONFIG REQUIRED)
find_package(Python 3.7 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(scipp 0.11 REQUIRED)

# MP : Parallel compile, add before any targets so all use it
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:-MP>)

include(GoogleTest)
include(GenerateExportHeader)
include(IPO)
include(scipp-install)

option(WITH_CTEST "Enable ctest integration of tests" ON)
option(DYNAMIC_LIB "Build shared libraries" ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Optimization flags
if(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --param inline-unit-growth=50")
endif()

# Get version from git-describe
execute_process(
  COMMAND git describe --tag
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  RESULT_VARIABLE HAVE_GIT_VERSION_INFO
  OUTPUT_VARIABLE GIT_VERSION_INFO OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(${HAVE_GIT_VERSION_INFO} EQUAL 0)
  message(STATUS "Got version from Git: ${GIT_VERSION_INFO}")
  add_definitions(-DSCIPPNEUTRON_VERSION="${GIT_VERSION_INFO}")
endif()

if(NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
  add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wunused
    -Wshadow
    -Winit-self
    -Wpointer-arith
    -Woverloaded-virtual
    -Wold-style-cast
    -Wcast-qual
    -Wcast-align
  )
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Woverloaded-virtual>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-operator-names>
  )
endif(NOT WIN32)
# This is currently causing to many warnings, reenable when appropriate.
# add_compile_options ( -Wconversion )

# permissive- : Std compliant parsing, warnings (W3) set by CMake defaults
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:-permissive->)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Custom target for building tests. all excludes tests by default
add_custom_target(all-tests)

option(COVERAGE "Enable code coverage run." OFF)
if(COVERAGE)
  include(CodeCoverage)
  configure_file(
    ${CMAKE_SOURCE_DIR}/tools/run-tests.sh.in
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/run-tests.sh" COPYONLY
  )
  set(COVERAGE_EXCLUDES "/*/include/scipp/*" "/usr/*" "/*conan/data*"
                        "/*miniconda*"
  )
  setup_target_for_coverage_lcov(
    NAME
    coverage
    EXECUTABLE
    sh
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/run-tests.sh"
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    "${CMAKE_SOURCE_DIR}/tests"
    DEPENDENCIES
    install-scippneutron
    all-tests
  )
endif()

if(NOT APPLE)
  set(CMAKE_INSTALL_RPATH $ORIGIN)
else()
  set(CMAKE_INSTALL_RPATH @loader_path)
endif()

set(EXPORT_NAME ${PROJECT_NAME}-targets)

if(DYNAMIC_LIB)
  install(
    EXPORT ${EXPORT_NAME}
    DESTINATION "lib/cmake/${PROJECT_NAME}"
    NAMESPACE ${PROJECT_NAME}::
  )
  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
  )
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
    VERSION "${CMAKE_PROJECT_VERSION}"
    COMPATIBILITY SameMinorVersion # switch to SameMajorVersion after 1.0
                                   # release
  )
  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
    DESTINATION lib/cmake/${PROJECT_NAME}
  )
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.conan-recipes"
          DESTINATION lib/cmake/${PROJECT_NAME}
  )
endif(DYNAMIC_LIB)

option(FULL_BUILD
       "Enable full build where the 'all' target includes tests and benchmarks."
       OFF
)

enable_testing()

add_subdirectory(neutron)
add_subdirectory(python)
