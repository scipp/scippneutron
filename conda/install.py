from os.path import join
import argparse
import shutil

parser = argparse.ArgumentParser(
    description='Move the install target to finalize conda-build')
parser.add_argument('--platform', default=None)
parser.add_argument('--source', default='')
parser.add_argument('--destination', default='')

if __name__ == '__main__':

    if 'windows' in args.platform.lower():
        lib_dest = 'lib'
        bin_src = 'bin'
        lib_src = 'Lib'
        inc_src = 'include'
    else:
        lib_dest = join('lib', 'python*')
        bin_src = None
        lib_src = 'lib'
        inc_src = 'include'

    shutil.move(join(args.source, 'scippneutron'),
                join(args.destination, lib_dest))
    if bin_src is not None:
        shutil.move(join(args.source, bin_src, 'scippneutron*.dll'),
                    join(args.destination, bin_src))
    shutil.move(join(args.source, lib_src, '*scippneutron*'),
                join(args.destination, lib_src))
    shutil.move(join(args.source, lib_src, 'cmake', 'scippneutron'),
                join(args.destination, lib_src, 'cmake'))
    shutil.move(join(args.source, inc_src, 'scippneutron*'),
                join(args.destination, inc_src))
    shutil.move(join(args.source, inc_src, 'scipp', 'neutron'),
                join(args.destination, inc_src, 'scipp'))
