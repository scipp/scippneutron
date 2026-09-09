import scipp as sc

from scippneutron.normalization import compute_single_crystal_norm


def test_can_call_compute_single_crystal_norm() -> None:
    """
    Only test that the function can be called and returns data of the correct shape.
    Detailed tests are in scippneutron-algorithms.
    """
    trajectory_start = sc.array(
        dims=['pixel', 'q-e'],
        values=[[0.1, 0.2, 0.0, 1.0], [0.9, 0.0, 0.5, 1.5]],
        unit='1/Å',
    )
    trajectory_stop = sc.array(
        dims=['pixel', 'q-e'],
        values=[[0.5, 0.2, 0.4, 0.9], [0.8, -0.1, 0.4, 1.4]],
        unit='1/Å',
    )
    solid_angle = sc.array(dims=["pixel"], values=[1.2, 0.8])
    grid = (
        sc.linspace('h', -0.2, 0.8, 10, unit='1/Å'),
        sc.linspace("k", -0.8, 0.8, 5, unit="1/Å"),
        sc.linspace("l", -0.6, 0.5, 4, unit="1/Å"),
        sc.linspace("energy_transfer", -5.5, 1.2, 7, unit="meV"),
    )
    incident_energy = sc.scalar(0.9, unit='meV')

    norm = compute_single_crystal_norm(
        trajectory_start=trajectory_start,
        trajectory_stop=trajectory_stop,
        solid_angle=solid_angle,
        grid=grid,
        incident_energy=incident_energy,
        n_threads=1,
    )

    assert norm.sizes == {'h': 9, 'k': 4, 'l': 3, 'energy_transfer': 6}
    assert norm.coords["h"].sizes == {"h": 10}
    assert norm.coords["k"].sizes == {"k": 5}
    assert norm.coords["l"].sizes == {"l": 4}
    assert norm.coords["energy_transfer"].sizes == {"energy_transfer": 7}
    assert sc.any(norm.data > sc.scalar(0.0, unit='1/meV'))
    assert sc.all(norm.data >= sc.scalar(0.0, unit='1/meV'))
