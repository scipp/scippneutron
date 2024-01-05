# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import scipp as sc

from ..mantid import load_with_mantid
from ._chopper_mockup import chopper_mockup

_version = '5'

__all__ = ['chopper_mockup', 'tutorial_dense_data', 'tutorial_event_data', 'get_path']


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache('scippneutron'),
        env='SCIPPNEUTRON_DATA_DIR',
        retry_if_failed=3,
        base_url='https://public.esss.dk/groups/scipp/scippneutron/{version}/',
        version=_version,
        registry={
            'bigfake.nxs': 'md5:ebf73017ddd03cee2a14a887c439bf35',
            'iris26176_graphite002_sqw.nxs': 'md5:7ea63f9137602b7e9b604fe30f0c6ec2',
            'loki-at-larmor.hdf5': 'md5:9c93f36fa2e94a7b63f2b946705e74f3',
            'loki-at-larmor-filtered.hdf5': 'md5:2789d0384339d1e130b86a44f22d11c6',
            'powder-event.h5': 'md5:38f7285b9814d00b5c5b3e63d4455e55',
            'powder-event-filtered.h5': 'md5:2eece38997aa800ec96e4105f85c11bd',
            'mcstas_sans.h5': 'md5:a608255dd757905490882eb35e209d12',
            'CNCS_51936_event.nxs': 'md5:5ba401e489260a44374b5be12b780911',
            'GEM40979.raw': 'md5:6df0f1c2fc472af200eec43762e9a874',
            'PG3_4844_calibration.h5': 'md5:39005d7b4befcafa173f00645549e2b4',
            'PG3_4844_event.nxs': 'md5:d5ae38871d0a09a28ae01f85d969de1e',
            'PG3_4866_event.nxs': 'md5:3d543bc6a646e622b3f4542bc3435e7e',
            'PG3_4871_event.nxs': 'md5:a3d0edcb36ab8e9e3342cd8a4440b779',
            'WISH00016748.raw': 'md5:37ecc6f99662b57e405ed967bdc068af',
        },
    )


_pooch = _make_pooch()


def bigfake():
    return get_path('bigfake.nxs')


def tutorial_dense_data():
    # This file was produced from loki-at-larmor.hdf5 by converting the data into a
    # data group. But not all attributes were copied.
    return sc.io.load_hdf5(_pooch.fetch('loki-at-larmor-filtered.hdf5'))


def tutorial_event_data():
    # This file was produced from powder-event.h5 by converting the data into a
    # data group. But not all attributes were copied.
    return sc.io.load_hdf5(_pooch.fetch('powder-event-filtered.h5'))


def powder_sample():
    return load_with_mantid(_pooch.fetch('PG3_4844_event.nxs'))


def powder_calibration():
    return sc.io.load_hdf5(_pooch.fetch('PG3_4844_calibration.h5'))


def get_path(name: str) -> str:
    """
    Return the path to a data file bundled with scippneutron.

    This function only works with example data and cannot handle
    paths to custom files.
    """
    return _pooch.fetch(name)
