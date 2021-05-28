import os
import argparse
import shutil
import glob

parser = argparse.ArgumentParser(
    description='Move the install target to finalize conda-build')
parser.add_argument('--platform', default='')
parser.add_argument('--source', default='')
parser.add_argument('--destination', default='')
args = parser.parse_args()


def move(src, dst):
    src = os.path.join(args.source, *src)
    dst = os.path.join(args.destination, *dst)
    # Note: do not check for '*' in dst
    if '*' in src:
        for f in glob.glob(src):
            shutil.move(src, dst)
    else:
        shutil.move(src, dst)


if __name__ == '__main__':

    os.write(1, "{}\n".format(args).encode())

    if 'windows' in args.platform.lower():
        lib_dest = 'lib'
        bin_src = 'bin'
        lib_src = 'Lib'
        inc_src = 'include'
    else:
        lib_dest = os.path.join('lib', 'python*')
        bin_src = None
        lib_src = 'lib'
        inc_src = 'include'

    move(['scippneutron'], [lib_dest])
    if bin_src is not None:
        move([bin_src, 'scippneutron*.dll'], [bin_src])
    move([lib_src, '*scippneutron*'], [lib_src])
    move([lib_src, 'cmake', 'scippneutron'], [lib_src, 'cmake'])
    move([inc_src, 'scippneutron*'], [inc_src])
    move([inc_src, 'scipp', 'neutron'], [inc_src, 'scipp'])
