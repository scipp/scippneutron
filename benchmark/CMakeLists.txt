# ~~~
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# ~~~

add_custom_target(all-benchmarks)

add_executable(
  neutron_convert_benchmark EXCLUDE_FROM_ALL neutron_convert_benchmark.cpp
)
add_dependencies(all-benchmarks neutron_convert_benchmark)
target_link_libraries(
  neutron_convert_benchmark LINK_PRIVATE benchmark::benchmark scippneutron
)
