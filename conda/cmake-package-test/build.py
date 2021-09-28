import os
import sys
import subprocess

shell = sys.platform == 'win32'

os.chdir(os.path.dirname(os.path.realpath(__file__)))
build_dir = os.path.relpath('build')
os.makedirs(build_dir)
os.chdir(build_dir)

subprocess.check_call([
    'cmake', f'-DPKG_VERSION={os.environ["PKG_VERSION"]}', '-DCMAKE_BUILD_TYPE=Release',
    '..'
],
                      stderr=subprocess.STDOUT,
                      shell=shell)
subprocess.check_call(['cmake', '--build', '.'], stderr=subprocess.STDOUT, shell=shell)
