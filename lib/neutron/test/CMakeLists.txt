# ~~~
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# ~~~
set(TARGET_NAME "scippneutron-test")
add_dependencies(all-tests ${TARGET_NAME})
add_executable(${TARGET_NAME} beamline_test.cpp constants_test.cpp)
target_link_libraries(${TARGET_NAME} LINK_PRIVATE scippneutron GTest::GTest)
set_property(
  TARGET ${TARGET_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION
                                 ${INTERPROCEDURAL_OPTIMIZATION_TESTS}
)
set_property(
  TARGET ${TARGET_NAME} PROPERTY EXCLUDE_FROM_ALL $<NOT:$<BOOL:${FULL_BUILD}>>
)
scipp_test(${TARGET_NAME} neutron)

if(WIN32)
  # Cmake uses ; as a separator and won't set the path correctly unless we
  # manually escape the path first.
  string(REPLACE ";" "\\;" PATH_STRING "$ENV{PATH}")
  set_tests_properties(
    scippneutron-test
    PROPERTIES
      ENVIRONMENT
      "PATH=${PATH_STRING}\;$ENV{CONDA_PREFIX}\\Lib\\site-packages\\scipp"
  )
endif(WIN32)
