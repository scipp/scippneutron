# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import scippbuildtools as sbt


def main(prefix, build_dir, source_dir, caching):
    builder = sbt.CppBuilder(prefix=prefix,
                             build_dir=build_dir,
                             source_dir=source_dir,
                             caching=caching)

    builder.cmake_configure()
    builder.enter_build_dir()
    builder.cmake_run()
    builder.cmake_build(['all-benchmarks', 'all-tests', 'install'])
    builder.run_cpp_tests(['scippneutron-test'])


if __name__ == '__main__':

    args = sbt.cpp_argument_parser().parse_known_args()[0]

    main(prefix=args.prefix,
         build_dir=args.build_dir,
         source_dir=args.source_dir,
         caching=args.caching)
