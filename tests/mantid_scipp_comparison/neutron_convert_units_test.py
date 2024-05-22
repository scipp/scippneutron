# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

import pytest
import scipp as sc

import scippneutron as scn

try:
    import mantid.kernel as kernel
    import mantid.simpleapi as sapi
except ImportError:
    pytestmark = pytest.mark.skip('Mantid framework is unavailable')
    sapi = None
    kernel = None


def mantid_coord(scipp_coord: str) -> str:
    return {
        'tof': 'TOF',
        'wavelength': 'Wavelength',
        'dspacing': 'dSpacing',
        'energy': 'Energy',
        'energy_transfer': 'DeltaE',
    }[scipp_coord]


def make_workspace(
    coord: str, emode: str = 'Elastic', efixed: sc.Variable | None = None
) -> 'sapi.Workspace':
    ws = sapi.CreateSampleWorkspace(XMin=1000, NumBanks=1, StoreInADS=False)
    # Crop out spectra index 0 as has two_theta=0, gives inf d-spacing
    ws = sapi.CropWorkspace(ws, StartWorkspaceIndex=1, StoreInADS=False)
    ws = sapi.ConvertUnits(
        InputWorkspace=ws,
        Target=mantid_coord(coord),
        EMode=emode,
        EFixed=efixed.value if efixed is not None else None,
        StoreInADS=False,
    )  # start in origin units
    ws.mutableRun().addProperty(
        'deltaE-mode', kernel.StringPropertyWithValue('deltaE-mode', emode), '', True
    )
    if efixed is not None:
        ws.mutableRun().addProperty(
            'Ei',
            kernel.FloatPropertyWithValue('Ei', efixed.value),
            str(efixed.unit),
            False,
        )
    return ws


def mantid_convert_units(
    ws: 'sapi.Workspace',
    target: str,
    emode: str = 'Elastic',
    efixed: sc.Variable | None = None,
) -> sc.DataArray:
    out_ws = sapi.ConvertUnits(
        InputWorkspace=ws,
        Target=mantid_coord(target),
        EMode=emode,
        EFixed=efixed.value if efixed is not None else None,
        StoreInADS=False,
    )
    out = scn.mantid.from_mantid(out_ws)['data']
    # broadcast to circumvent common-bins conversion in from_mantid
    spec_shape = out.coords['spectrum'].shape
    out.coords[target] = (
        sc.ones(dims=['spectrum'], shape=spec_shape) * out.coords[target]
    )
    return out


def test_mantid_convert_tof_to_wavelength():
    in_ws = make_workspace('tof')
    out_mantid = mantid_convert_units(in_ws, 'wavelength')

    in_da = scn.mantid.from_mantid(in_ws)['data']
    out_scipp = scn.convert(data=in_da, origin='tof', target='wavelength', scatter=True)

    assert sc.allclose(
        out_scipp.coords['wavelength'],
        out_mantid.coords['wavelength'],
        rtol=1e-8 * sc.units.one,
    )
    assert sc.identical(out_scipp.coords['spectrum'], out_mantid.coords['spectrum'])


def test_mantid_convert_tof_to_dspacing():
    in_ws = make_workspace('tof')
    out_mantid = mantid_convert_units(in_ws, 'dspacing')

    in_da = scn.mantid.from_mantid(in_ws)['data']
    out_scipp = scn.convert(data=in_da, origin='tof', target='dspacing', scatter=True)

    assert sc.allclose(
        out_scipp.coords['dspacing'],
        out_mantid.coords['dspacing'],
        rtol=1e-8 * sc.units.one,
    )
    assert sc.identical(out_scipp.coords['spectrum'], out_mantid.coords['spectrum'])


def test_mantid_convert_tof_to_energy():
    in_ws = make_workspace('tof')
    out_mantid = mantid_convert_units(in_ws, 'energy')

    in_da = scn.mantid.from_mantid(in_ws)['data']
    out_scipp = scn.convert(data=in_da, origin='tof', target='energy', scatter=True)

    # Mantid reverses the order of the energy dim.
    mantid_energy = sc.empty_like(out_mantid.coords['energy'])
    assert mantid_energy.dims[1] == 'energy'
    mantid_energy.values = out_mantid.coords['energy'].values[..., ::-1]

    assert sc.allclose(
        out_scipp.coords['energy'], mantid_energy, rtol=1e-7 * sc.units.one
    )
    assert sc.identical(out_scipp.coords['spectrum'], out_mantid.coords['spectrum'])


def test_mantid_convert_tof_to_direct_energy_transfer():
    efixed = 1000.0 * sc.Unit('meV')
    in_ws = make_workspace('tof', emode='Direct', efixed=efixed)
    out_mantid = mantid_convert_units(
        in_ws, 'energy_transfer', emode='Direct', efixed=efixed
    )

    in_da = scn.mantid.from_mantid(in_ws)['data']
    out_scipp = scn.convert(
        data=in_da, origin='tof', target='energy_transfer', scatter=True
    )

    # The conversion consists of multiplications and additions, thus the relative error
    # changes with the inputs. In this case, small tof yields a large error due to
    # the 1/tof**2 factor in the conversion.
    # rtol is chosen to account for linearly changing tof in the input data.
    assert sc.allclose(
        out_scipp.coords['energy_transfer'],
        out_mantid.coords['energy_transfer'],
        rtol=sc.linspace(
            'energy_transfer',
            1e-6,
            1e-10,
            out_scipp.coords['energy_transfer'].sizes['energy_transfer'],
        ),
    )
    assert sc.identical(out_scipp.coords['spectrum'], out_mantid.coords['spectrum'])
