# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import os
from pathlib import Path
import scippbuildtools as sbt

if __name__ == '__main__':

    args = sbt.docs_argument_parser().parse_known_args()[0]

    docs_dir = str(Path(__file__).parent.absolute())

    builder = sbt.DocsBuilder(docs_dir=docs_dir,
                              prefix=args.prefix,
                              work_dir=args.work_dir,
                              data_dir=args.data_dir)

    if not args.no_setup:
        builder.download_test_data(tar_name="scippneutron.tar.gz")
        builder.make_mantid_config(
            content="\nusagereports.enabled=0\ndatasearch.directories={}\n"
            "logging.loggers.root.level=error\n".format(
                os.path.join(args.data_dir, "scippneutron")))

    builder.run_sphinx(builder=args.builder, docs_dir=docs_dir)
