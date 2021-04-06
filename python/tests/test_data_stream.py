from scippneutron.data_stream import _data_stream
from scippneutron._streaming_data_buffer import StreamedDataBuffer
import asyncio
import pytest
from typing import List
from streaming_data_types import serialise_ev42
import numpy as np


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


# Short time to use for buffer emit and data_stream interval in tests
# ensure they pass or fail fast!
SHORT_TEST_INTERVAL = 0.005  # 5 ms


@pytest.mark.asyncio
async def test_data_stream_returns_data_from_single_event_message():
    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, interval_s=SHORT_TEST_INTERVAL)
    consumers = [FakeConsumer()]
    time_of_flight = np.array([1., 2., 3.])
    detector_ids = np.array([4, 5, 6])
    fake_message = serialise_ev42("detector", 0, 0, time_of_flight,
                                  detector_ids)
    await buffer.new_data(fake_message)

    collected_data = None
    async for data in _data_stream(buffer,
                                   queue,
                                   consumers,
                                   interval_s=SHORT_TEST_INTERVAL):
        collected_data = data
        stop_consumers(consumers)

    assert collected_data is not None
    assert np.allclose(collected_data.coords['tof'].values, time_of_flight)


@pytest.mark.asyncio
async def test_data_stream_returns_data_from_multiple_event_messages():
    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, interval_s=SHORT_TEST_INTERVAL)
    consumers = [FakeConsumer()]
    first_tof = np.array([1., 2., 3.])
    first_detector_ids = np.array([4, 5, 6])
    first_fake_message = serialise_ev42("detector", 0, 0, first_tof,
                                        first_detector_ids)
    second_tof = np.array([1., 2., 3.])
    second_detector_ids = np.array([4, 5, 6])
    second_fake_message = serialise_ev42("detector", 0, 0, second_tof,
                                         second_detector_ids)
    await buffer.new_data(first_fake_message)
    await buffer.new_data(second_fake_message)

    collected_data = None
    async for data in _data_stream(buffer,
                                   queue,
                                   consumers,
                                   interval_s=SHORT_TEST_INTERVAL):
        collected_data = data
        stop_consumers(consumers)

    assert collected_data is not None
    expected_tofs = np.concatenate((first_tof, second_tof))
    assert np.allclose(collected_data.coords['tof'].values, expected_tofs)
