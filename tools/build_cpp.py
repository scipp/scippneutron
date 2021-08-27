# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import scippbuildtools as sbt


def main(**kwargs):
    builder = sbt.CppBuilder(**kwargs)
    builder.cmake_configure()
    builder.enter_build_dir()
    builder.cmake_run()
    builder.cmake_build(['all-benchmarks', 'all-tests', 'install'])
    builder.run_cpp_tests(['scippneutron-test'])


if __name__ == '__main__':

    args = sbt.cpp_argument_parser().parse_known_args()[0]
    # Convert Namespace object `args` to a dict with `vars(args)`
    main(**vars(args))
