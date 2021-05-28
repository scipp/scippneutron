# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

import os
import argparse
import urllib
import requests
import re
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description='Build doc pages with sphinx')
parser.add_argument('--prefix', default='build')
parser.add_argument('--work_dir', default='.doctrees')
parser.add_argument('--data_dir', default='data')

args = parser.parse_args()


def download_file(source, target):
    os.write(1, "Downloading: {}\n".format(source).encode())
    urllib.request.urlretrieve(source, target)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def download_multiple(remote_url, target_dir, extensions):
    """
    Generate file list by parsing the html source of the web server and search
    for links that include the relevant file extensions.
    Then download all the files in the list.
    """
    make_dir(target_dir)
    page_source = requests.get(remote_url).text
    data_files = []
    for ext in extensions:
        for f in re.findall(r'href=.*{}">'.format(ext), page_source):
            data_files.append(f.lstrip('href="').rstrip('">'))
    for f in data_files:
        target = os.path.join(target_dir, f)
        # Note that only checking if file exists won't download new versions of
        # files that are already on disk
        if not os.path.isfile(target):
            download_file(os.path.join(remote_url, f), target)


if __name__ == '__main__':

    # Download data files
    remote_url = "https://public.esss.dk/groups/scipp/scippneutron"
    target_dir = os.path.abspath(args.data_dir)
    extensions = [".nxs", ".h5", ".hdf5", ".raw"]
    download_multiple(remote_url, target_dir, extensions)

    # Create Mantid properties file so that it can find the data files
    home = str(Path.home())
    config_dir = os.path.join(home, ".mantid")
    make_dir(config_dir)
    properties_file = os.path.join(config_dir, "Mantid.user.properties")
    with open(properties_file, "a") as f:
        f.write("\nusagereports.enabled=0\ndatasearch.directories={}\n".format(
            target_dir))

    status = subprocess.check_call(
        ['sphinx-build', '-d', args.work_dir, '.', args.prefix],
        stderr=subprocess.STDOUT,
        shell=sys.platform == "win32")
