# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import os
from pathlib import Path
import scippbuildtools as sbt

if __name__ == '__main__':
    args, _ = sbt.docs_argument_parser().parse_known_args()
    docs_dir = str(Path(__file__).parent.absolute())
    # Convert Namespace object `args` to a dict with `vars(args)`
    builder = sbt.DocsBuilder(docs_dir=docs_dir, **vars(args))

    if not args.no_setup:
        builder.download_test_data(tar_name="scippneutron.tar.gz")
        builder.make_mantid_config(
            content="\nusagereports.enabled=0\ndatasearch.directories={}\n"
            "logging.loggers.root.level=error\n".format(
                os.path.join(builder._data_dir, "scippneutron")))

    builder.run_sphinx(builder=args.builder, docs_dir=docs_dir)
