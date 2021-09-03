# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
try:
    from scippbuildtools.sphinxconf import *  # noqa: E402, F401, F403
except ImportError:
    pass

project = u'scippneutron'

nbsphinx_prolog = nbsphinx_prolog.replace("XXXX", "scippneutron")  # noqa: F405
