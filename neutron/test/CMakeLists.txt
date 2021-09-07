# ~~~
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# ~~~
set(TARGET_NAME "scippneutron-test")
add_dependencies(all-tests ${TARGET_NAME})
add_executable(
  ${TARGET_NAME} EXCLUDE_FROM_ALL beamline_test.cpp constants_test.cpp
  convert_test.cpp
)
target_link_libraries(${TARGET_NAME} LINK_PRIVATE scippneutron GTest::GTest)
set_property(
  TARGET ${TARGET_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION
                                 ${INTERPROCEDURAL_OPTIMIZATION_TESTS}
)
if(${WITH_CTEST})
  gtest_discover_tests(${TARGET_NAME} TEST_PREFIX bin/)
endif()
