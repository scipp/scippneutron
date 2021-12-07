# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from skbuild import setup
from setuptools import find_packages


def get_version():
    import subprocess
    return subprocess.run(['git', 'describe', '--tags', '--abbrev=0'],
                          stdout=subprocess.PIPE).stdout.decode('utf8').strip()


def get_cmake_args():
    # Note: We do not specify '-DCMAKE_OSX_DEPLOYMENT_TARGET' here. It is set using the
    # MACOSX_DEPLOYMENT_TARGET environment variable in the github workflow. The reason
    # is that I am not sure if cibuildwheel uses this for anything else apart from
    # configuring the actual build.
    return []


setup(name='scippneutron',
      version=get_version(),
      description='Neutron scattering data processing based on scipp',
      author='Scipp contributors (https://github.com/scipp)',
      url='https://scipp.github.io/scippneutron',
      license='BSD-3-Clause',
      packages=find_packages(where="src"),
      package_dir={'': 'src'},
      cmake_args=get_cmake_args(),
      cmake_install_dir='src/scippneutron',
      include_package_data=True,
      python_requires='>=3.7',
      install_requires=[
          'scipp==0.10.0rc4',
      ],
      extras_require={
          "test": ["pytest"],
      })
