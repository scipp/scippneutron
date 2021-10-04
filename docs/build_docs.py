# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

import os
import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description='Build doc pages with sphinx')
parser.add_argument('--prefix', default='build')
parser.add_argument('--work_dir', default='.doctrees')
parser.add_argument('--builder', default='html')


def _get_abs_path(path, root):
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(root, path)


if __name__ == '__main__':

    args = parser.parse_known_args()[0]

    docs_dir = Path(__file__).parent.absolute()
    work_dir = _get_abs_path(path=args.work_dir, root=docs_dir)
    prefix = _get_abs_path(path=args.prefix, root=docs_dir)

    # Build the docs with sphinx-build
    status = subprocess.check_call(
        ['sphinx-build', '-b', args.builder, '-d', work_dir, docs_dir, prefix],
        stderr=subprocess.STDOUT,
        shell=sys.platform == "win32")
