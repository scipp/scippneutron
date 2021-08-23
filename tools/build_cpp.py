# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import scippbuildtools as sbt

if __name__ == '__main__':
    args = sbt.make_argument_parser().parse_args()

    builder = sbt.CppBuilder(prefix=args.prefix,
                             build_dir=args.build_dir,
                             source_dir=args.source_dir,
                             caching=args.caching)

    builder.cmake_configure()
    builder.enter_build_dir()
    builder.cmake_run()
    builder.cmake_build(['all-benchmarks', 'all-tests', 'install'])
    builder.run_cpp_tests(['scippneutron-test'])
