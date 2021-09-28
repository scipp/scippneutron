import os
import sys
import subprocess
import tempfile

shell = sys.platform == 'win32'

with tempfile.TemporaryDirectory() as build_dir:
    with tempfile.TemporaryDirectory() as work_dir:
        subprocess.check_call([
            'python', 'build_docs.py', '--builder=html', f'--prefix={build_dir}',
            f'--work_dir={work_dir}'
        ],
                              stderr=subprocess.STDOUT,
                              shell=shell)
    with tempfile.TemporaryDirectory() as work_dir:
        subprocess.check_call([
            'python', 'build_docs.py', '--builder=doctest', f'--prefix={build_dir}',
            f'--work_dir={work_dir}', '--no-setup'
        ],
                              stderr=subprocess.STDOUT,
                              shell=shell)
