import datetime

import pytest
import scipp as sc
import asyncio
from typing import List
import numpy as np
from .nexus_helpers import NexusBuilder

try:
    import streaming_data_types  # noqa: F401
    import confluent_kafka  # noqa: F401
    from scippneutron.data_stream import _data_stream  # noqa: E402
    from scippneutron._streaming_data_buffer import \
        StreamedDataBuffer  # noqa: E402
    from streaming_data_types.eventdata_ev42 import \
        serialise_ev42  # noqa: E402
    from streaming_data_types.run_start_pl72 import (serialise_pl72,
                                                     deserialise_pl72,
                                                     RunStartInfo)
except ImportError:
    pytest.skip("Kafka or Serialisation module is unavailable",
                allow_module_level=True)


class FakeConsumer:
    """
    Use in place of KafkaConsumer to avoid having to do
    network IO in unit tests. Does not need to supply
    fake messages as the new_data method on the
    StreamedDataBuffer can be called manually instead.
    """
    def __init__(self):
        self.stopped = True

    def start(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


def stop_consumers(consumers: List[FakeConsumer]):
    for consumer in consumers:
        consumer.stop()


def get_run_start_message_no_streams(topic: str, broker: str) -> RunStartInfo:
    """
    The real implementation finds the last run start
    message in the given topic.
    For tests we'll construct a run start message and
    return it.
    """
    builder = NexusBuilder()
    builder.add_instrument("DATA_STREAM_TEST")
    message = serialise_pl72("",
                             "",
                             datetime.datetime.now(),
                             nexus_structure=builder.json_string)
    return deserialise_pl72(message)


# Short time to use for buffer emit and data_stream interval in tests
# pass or fail fast!
SHORT_TEST_INTERVAL = 1. * sc.Unit('milliseconds')
# Small buffer of 20 events is sufficient for the tests
TEST_BUFFER_SIZE = 20


@pytest.mark.asyncio
async def test_data_stream_returns_data_from_single_event_message():
    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, TEST_BUFFER_SIZE, SHORT_TEST_INTERVAL)
    consumers = [FakeConsumer()]
    time_of_flight = np.array([1., 2., 3.])
    detector_ids = np.array([4, 5, 6])
    test_message = serialise_ev42("detector", 0, 0, time_of_flight,
                                  detector_ids)
    await buffer.new_data(test_message)

    async for data in _data_stream(
            buffer,
            queue,
            consumers,  # type: ignore
            SHORT_TEST_INTERVAL,
            ""):
        assert np.allclose(data.coords['tof'].values, time_of_flight)

        # Cause the data_stream generator to stop and exit the "async for"
        stop_consumers(consumers)


@pytest.mark.asyncio
async def test_data_stream_returns_data_from_multiple_event_messages():
    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, TEST_BUFFER_SIZE, SHORT_TEST_INTERVAL)
    consumers = [FakeConsumer()]
    first_tof = np.array([1., 2., 3.])
    first_detector_ids = np.array([4, 5, 6])
    first_test_message = serialise_ev42("detector", 0, 0, first_tof,
                                        first_detector_ids)
    second_tof = np.array([1., 2., 3.])
    second_detector_ids = np.array([4, 5, 6])
    second_test_message = serialise_ev42("detector", 0, 0, second_tof,
                                         second_detector_ids)
    await buffer.new_data(first_test_message)
    await buffer.new_data(second_test_message)

    async for data in _data_stream(
            buffer,
            queue,
            consumers,  # type: ignore
            SHORT_TEST_INTERVAL,
            ""):
        expected_tofs = np.concatenate((first_tof, second_tof))
        assert np.allclose(data.coords['tof'].values, expected_tofs)
        expected_ids = np.concatenate(
            (first_detector_ids, second_detector_ids))
        assert np.array_equal(data.coords['detector_id'].values, expected_ids)

        stop_consumers(consumers)


@pytest.mark.asyncio
async def test_warn_on_data_emit_if_unrecognised_message_was_encountered():
    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, TEST_BUFFER_SIZE, SHORT_TEST_INTERVAL)
    # First 4 bytes of the message payload are the FlatBuffer schema identifier
    # "abcd" does not correspond to a FlatBuffer schema for data
    # that scipp is interested in
    test_message = b"abcd0000"
    await buffer.new_data(test_message)

    with pytest.warns(UserWarning):
        await buffer._emit_data()


@pytest.mark.asyncio
async def test_warn_on_buffer_size_exceeded_by_single_message():
    queue = asyncio.Queue()
    buffer_size_2_events = 2
    buffer = StreamedDataBuffer(queue,
                                buffer_size=buffer_size_2_events,
                                interval=SHORT_TEST_INTERVAL)
    time_of_flight = np.array([1., 2., 3.])
    detector_ids = np.array([4, 5, 6])
    test_message = serialise_ev42("detector", 0, 0, time_of_flight,
                                  detector_ids)

    # User is warned to try again with a larger buffer size,
    # and informed what message size was encountered
    with pytest.warns(UserWarning):
        await buffer.new_data(test_message)


@pytest.mark.asyncio
async def test_buffer_size_exceeded_by_messages_causes_early_data_emit():
    queue = asyncio.Queue()
    buffer_size_5_events = 5
    buffer = StreamedDataBuffer(queue,
                                buffer_size=buffer_size_5_events,
                                interval=SHORT_TEST_INTERVAL)
    first_tof = np.array([1., 2., 3.])
    first_detector_ids = np.array([4, 5, 6])
    first_test_message = serialise_ev42("detector", 0, 0, first_tof,
                                        first_detector_ids)
    second_tof = np.array([1., 2., 3.])
    second_detector_ids = np.array([4, 5, 6])
    second_test_message = serialise_ev42("detector", 0, 0, second_tof,
                                         second_detector_ids)

    with pytest.warns(None) as record_warnings:
        await buffer.new_data(first_test_message)
        assert len(
            record_warnings
        ) == 0, "Expect no warning from first message as events " \
                "fit in buffer"

    assert queue.empty()
    await buffer.new_data(second_test_message)
    assert not queue.empty(), "Expect data to have been emitted to " \
                              "queue as buffer size was exceeded"


@pytest.mark.asyncio
async def test_data_are_loaded_from_run_start_message():
    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, TEST_BUFFER_SIZE, SHORT_TEST_INTERVAL)
    consumers = []
    run_info_topic = "fake_topic"
    reached_assert = False
    async for data in _data_stream(
            buffer,
            queue,
            consumers,  # type: ignore
            SHORT_TEST_INTERVAL,
            "broker_address",
            run_info_topic,
            get_run_start_message_no_streams):
        assert data["instrument_name"].value == "DATA_STREAM_TEST"
        reached_assert = True
    assert reached_assert
