# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

__all__ = ['get_scippnexus_path']


def _make_scippnexus_pooch():
    """Files from scippnexus.
    Uses the same setup in order to share files."""
    import pooch

    return pooch.create(
        path=pooch.os_cache('scippnexus-externalfile'),
        env='SCIPPNEXUS_DATA_DIR',
        retry_if_failed=3,
        base_url='login.esss.dk:/mnt/groupdata/scipp/testdata/scippnexus/',
        registry={
            '2023/BIFROST_873855_00000015.hdf': 'md5:eb180b09d265c308e81c4a4885662bbd',
        },
    )


_scippnexus_pooch = _make_scippnexus_pooch()


def sshdownloader(url, output_file, pooch):
    from subprocess import call

    cmd = ['scp', f'{url}', f'{output_file}']
    call(cmd)


def get_scippnexus_path(name: str) -> str:
    """
    Get path of file "downloaded" via SSH from login.esss.dk.

    You must have setup SSH agent for passwordless login for this to work.
    """
    return _scippnexus_pooch.fetch(name, downloader=sshdownloader)
