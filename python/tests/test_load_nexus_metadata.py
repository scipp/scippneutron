from .nexus_helpers import (
    NexusBuilder,
    Log,
    Sample,
    Source,
    Monitor,
    Chopper,
)
import numpy as np
import scipp as sc
from scippneutron import load_nexus_monitors, \
    load_nexus_instrument_name, load_nexus_disk_chopper, load_nexus_sample, \
    load_nexus_source, load_nexus_start_and_end_time, load_nexus_title, load_nexus_logs


def test_load_log_metadata():
    builder = NexusBuilder()
    log_1 = Log("test_log", np.array([1.1, 2.2, 3.3]), np.array([4.4, 5.5, 6.6]))
    log_2 = Log("test_log_2", np.array([123, 253, 756]), np.array([246, 1235, 2369]))
    builder.add_log(log_1)
    builder.add_log(log_2)

    with builder.file() as file:
        loaded = load_nexus_logs(file)

    assert np.allclose(loaded[log_1.name].values.data.values, log_1.value)
    assert np.allclose(loaded[log_1.name].values.coords['time'].values, log_1.time)
    assert np.array_equal(loaded[log_2.name].values.data.values, log_2.value)
    assert np.array_equal(loaded[log_2.name].values.coords['time'].values, log_2.time)


def test_load_title_metadata():
    builder = NexusBuilder()
    builder.add_title("my_experiment_title")

    with builder.file() as file:
        loaded = load_nexus_title(file)

    assert sc.identical(loaded["experiment_title"], sc.scalar("my_experiment_title"))


def test_load_start_and_end_time_metadata():
    builder = NexusBuilder()
    builder.add_run_start_time("1980-01-01T06:00:00Z")
    builder.add_run_end_time("1990-01-01T06:00:00Z")

    with builder.file() as file:
        loaded = load_nexus_start_and_end_time(file)

    assert sc.identical(loaded["start_time"], sc.scalar("1980-01-01T06:00:00Z"))
    assert sc.identical(loaded["end_time"], sc.scalar("1990-01-01T06:00:00Z"))


def test_load_nexus_source_metadata():
    builder = NexusBuilder()
    builder.add_source(Source(name="source_1", distance=5.0, distance_units="m"))

    with builder.file() as file:
        loaded = load_nexus_source(file)

    assert sc.identical(loaded["source_position"],
                        sc.vector(value=[0, 0, 5], unit=sc.units.m))


def test_load_nexus_sample_metadata():
    builder = NexusBuilder()
    builder.add_sample(Sample(name="sample_1", distance=3.0, distance_units="m"))

    with builder.file() as file:
        loaded = load_nexus_sample(file)

    assert sc.identical(loaded["sample_position"],
                        sc.vector(value=[0, 0, 3], unit=sc.units.m))


def test_load_nexus_disk_chopper_metadata():
    builder = NexusBuilder()
    builder.add_chopper(Chopper(name="chopper_1", distance=8.0, rotation_speed=300))

    with builder.file() as file:
        loaded = load_nexus_disk_chopper(file)

    assert sc.identical(
        loaded["chopper_1"],
        sc.DataArray(data=sc.scalar("chopper_1"),
                     coords={},
                     attrs={
                         "distance": sc.scalar(8.0),
                         "rotation_speed": sc.scalar(300, dtype=sc.dtype.int32),
                     }))


def test_load_nexus_inst_name_metadata():
    builder = NexusBuilder()
    builder.add_instrument("my_instrument_name")

    with builder.file() as file:
        loaded = load_nexus_instrument_name(file)

    assert sc.identical(loaded["instrument_name"], sc.scalar("my_instrument_name"))


def test_load_monitor_metadata():
    builder = NexusBuilder()
    builder.add_monitor(
        Monitor(
            name="monitor1",
            data=np.ones(shape=(2, 4, 6), dtype="float64"),
            axes=[
                ("event_index", np.arange(3, 5, dtype="float64")),
                ("period_index", np.arange(3, 7, dtype="float64")),
                ("time_of_flight", np.arange(3, 9, dtype="float64")),
            ],
        ))

    with builder.file() as file:
        loaded = load_nexus_monitors(file)

    assert sc.identical(
        loaded["monitor1"].values,
        sc.DataArray(data=sc.ones(
            dims=["event_index", "period_index", "time_of_flight"],
            shape=(2, 4, 6),
            dtype=sc.dtype.float64),
                     coords={
                         "event_index":
                         sc.Variable(dims=["event_index"],
                                     values=np.arange(3, 5, dtype="float64")),
                         "period_index":
                         sc.Variable(dims=["period_index"],
                                     values=np.arange(3, 7, dtype="float64")),
                         "time_of_flight":
                         sc.Variable(dims=["time_of_flight"],
                                     values=np.arange(3, 9, dtype="float64")),
                     }))
