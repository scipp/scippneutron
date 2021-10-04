# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import scipp as sc

from ..mantid import load

_version = '1'

__all__ = ['tutorial_dense_data', 'tutorial_event_data']


def _make_pooch():
    import pooch
    return pooch.create(
        path=pooch.os_cache('scippneutron'),
        base_url='https://public.esss.dk/groups/scipp/scippneutron/{version}/',
        version=_version,
        registry={
            'powder-event.h5': 'md5:b8ad26eb3efc2159687134a5396a2671',
            'loki-at-larmor.hdf5': 'md5:6691ef98406bd4d526e2131ece3c8d69',
            'GEM40979.raw': 'md5:6df0f1c2fc472af200eec43762e9a874',
            'PG3_4844_event.nxs': 'md5:d5ae38871d0a09a28ae01f85d969de1e',
            'PG3_4866_event.nxs': 'md5:3d543bc6a646e622b3f4542bc3435e7e',
            'PG3_4871_event.nxs': 'md5:a3d0edcb36ab8e9e3342cd8a4440b779',
        })


_pooch = _make_pooch()


def tutorial_dense_data():
    return sc.io.open_hdf5(_pooch.fetch('loki-at-larmor.hdf5'))


def tutorial_event_data():
    return sc.io.open_hdf5(_pooch.fetch('powder-event.h5'))


def example_data(name, *args, **kwargs):
    return load(_pooch.fetch(name), *args, **kwargs)
