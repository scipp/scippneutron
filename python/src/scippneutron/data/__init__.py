# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import scipp as sc
from .._scippneutron import __version__

_version = __version__.split('-')[0]


def _make_pooch():
    import pooch
    return pooch.create(
        path=pooch.os_cache('scippneutron'),
        base_url='https://public.esss.dk/groups/scipp/scippneutron/{version}/',
        version=_version,
        registry={
            'powder-event.h5': 'md5:b8ad26eb3efc2159687134a5396a2671',
            'loki-at-larmor.hdf5': 'md5:6691ef98406bd4d526e2131ece3c8d69',
        })


_pooch = _make_pooch()


def tutorial_dense_data():
    return sc.io.open_hdf5(_pooch.fetch('loki-at-larmor.hdf5'))


def tutorial_event_data():
    return sc.io.open_hdf5(_pooch.fetch('powder-event.h5'))
