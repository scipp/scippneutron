import os
import argparse
# import urllib
# import requests
# import re
# from pathlib import Path
import shutil
import subprocess
import multiprocessing
import sys

parser = argparse.ArgumentParser(description='Build C++ library and run tests')
parser.add_argument('--prefix', default='')
parser.add_argument('--osxversion', default='')
parser.add_argument('--build_dir', default='build')

args = parser.parse_args()


def run_command(cmd, shell):
    """
    Run a command (supplied as a list) using subprocess.check_call
    """
    os.write(1, "{}\n".format(' '.join(cmd)).encode())
    return subprocess.check_call(cmd, stderr=subprocess.STDOUT, shell=shell)


if __name__ == '__main__':

    platform = sys.platform
    print("PLATFORM IS:", platform)

    # Default options
    shell = False
    ncores = str(multiprocessing.cpu_count())
    parallel_flag = '-j'
    build_config = ''

    # Some flags use a syntax with a space separator instead of '='
    use_space = ['-G', '-A']

    cmake_flags = {
        '-G': 'Ninja',
        '-DPYTHON_EXECUTABLE': shutil.which("python"),
        '-DCMAKE_INSTALL_PREFIX': args.prefix,
        '-DWITH_CTEST': 'OFF',
        '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION': 'OFF'
    }

    if platform == 'linux':
        cmake_flags.update({'-DCMAKE_INTERPROCEDURAL_OPTIMIZATION': 'ON'})

    if platform == 'darwin':
        cmake_flags.update({
            '-DCMAKE_OSX_DEPLOYMENT_TARGET':
            args.osxversion,
            '-DCMAKE_OSX_SYSROOT':
            os.path.join('/Applications', 'Xcode.app', 'Contents', 'Developer',
                         'Platforms', 'MacOSX.platform', 'Developer', 'SDKs',
                         'MacOSX{}.sdk'.format(args.osxversion))
        })

    if platform == 'win32':
        cmake_flags.update({
            '-G': 'Visual Studio 16 2019',
            '-A': 'x64',
            '-DCMAKE_CXX_STANDARD': '20'
        })
        shell = True
        parallel_flag = '-- /m:'
        build_config = 'Release'

    # Additional flags for --build commands
    build_flags = []
    if len(build_config) > 0:
        build_flags += ['--config', build_config]
    build_flags += [parallel_flag + ncores]

    # Parse cmake flags
    flags_list = []
    for key, value in cmake_flags.items():
        if key in use_space:
            flags_list += [key, value]
        else:
            flags_list.append('{}={}'.format(key, value))

    if not os.path.exists(args.build_dir):
        os.makedirs(args.build_dir)
    os.chdir(args.build_dir)

    # Run cmake
    status = run_command(['cmake'] + flags_list + ['..'], shell=shell)

    # Show cmake settings
    status = run_command(['cmake', '-B', '.', '-S', '..', '-LA'], shell=shell)

    # Compile benchmarks
    status = run_command(
        ['cmake', '--build', '.', '--target', 'all-benchmarks'] + build_flags,
        shell=shell)

    # Compile C++ tests
    status = run_command(['cmake', '--build', '.', '--target', 'all-tests'] +
                         build_flags,
                         shell=shell)

    # Compile Python library
    status = run_command(['cmake', '--build', '.', '--target', 'install'] +
                         build_flags,
                         shell=shell)

    # Run C++ tests
    status = run_command(
        [os.path.join('bin', build_config, 'scippneutron-test')], shell=shell)
