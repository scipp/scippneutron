import datetime
import pytest
import scipp as sc
import asyncio
from typing import List, Tuple, Callable, Dict, Optional
import numpy as np
from .nexus_helpers import NexusBuilder, Stream
import dateutil.parser

try:
    import streaming_data_types  # noqa: F401
    from confluent_kafka import TopicPartition  # noqa: F401
    from scippneutron.data_stream import _data_stream, StartTime  # noqa: E402
    from scippneutron._streaming_data_buffer import \
        StreamedDataBuffer  # noqa: E402
    from streaming_data_types.eventdata_ev42 import \
        serialise_ev42  # noqa: E402
    from streaming_data_types.run_start_pl72 import serialise_pl72
    from scippneutron._streaming_consumer import RunStartError
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
    def __init__(self,
                 topic_partitions: Optional[List[TopicPartition]] = None,
                 conf: Optional[Dict] = None,
                 callback: Optional[Callable] = None,
                 stop_at_end_of_partition: Optional[bool] = None):
        self.stopped = True

    def start(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


def stop_consumers(consumers: List[FakeConsumer]):
    for consumer in consumers:
        consumer.stop()


class FakeMessage:
    def __init__(self, message_payload: bytes):
        self._message_payload = message_payload

    def value(self):
        return self._message_payload

    @staticmethod
    def error() -> bool:
        return False


class FakeQueryConsumer:
    def __init__(self,
                 instrument_name: str = "",
                 low_and_high_offset: Tuple[int, int] = (2, 10),
                 streams: List[Stream] = None,
                 start_time: Optional[str] = None):
        self._instrument_name = instrument_name
        self._low_and_high_offset = low_and_high_offset
        self._streams = streams
        self.queried_topics = []
        self.queried_timestamp = None
        self._start_time = start_time

    @staticmethod
    def assign(partitions: List[TopicPartition]):
        pass

    def get_watermark_offsets(self,
                              partition: TopicPartition) -> Tuple[int, int]:
        return self._low_and_high_offset

    def get_topic_partitions(self,
                             topic: str,
                             offset: int = -1) -> List[TopicPartition]:
        self.queried_topics.append(topic)
        return [TopicPartition(topic, partition=0, offset=offset)]

    def poll(self, timeout=2.) -> FakeMessage:
        builder = NexusBuilder()
        builder.add_instrument(self._instrument_name)
        if self._streams is not None:
            for stream in self._streams:
                builder.add_stream(stream)
        if self._start_time is not None:
            builder.add_dataset_at_path("/entry/start_time", self._start_time,
                                        {})
        return FakeMessage(
            serialise_pl72("",
                           "",
                           datetime.datetime.now(),
                           nexus_structure=builder.json_string))

    def seek(self, partition: TopicPartition):
        pass

    def offsets_for_times(self, partitions: List[TopicPartition]):
        self.queried_timestamp = partitions[0].offset
        return partitions


# Short time to use for buffer emit and data_stream interval in tests
# pass or fail fast!
SHORT_TEST_INTERVAL = 1. * sc.Unit('milliseconds')
# Small buffer of 20 events is sufficient for the tests
TEST_BUFFER_SIZE = 20


@pytest.mark.asyncio
async def test_data_stream_returns_data_from_single_event_message():
    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, TEST_BUFFER_SIZE, SHORT_TEST_INTERVAL)
    time_of_flight = np.array([1., 2., 3.])
    detector_ids = np.array([4, 5, 6])
    test_message = serialise_ev42("detector", 0, 0, time_of_flight,
                                  detector_ids)
    await buffer.new_data(test_message)

    reached_assert = False
    async for data in _data_stream(buffer,
                                   queue,
                                   "broker", ["topic"],
                                   SHORT_TEST_INTERVAL,
                                   query_consumer=FakeQueryConsumer(),
                                   consumer_type=FakeConsumer,
                                   max_iterations=1):
        assert np.allclose(data.coords['tof'].values, time_of_flight)
        reached_assert = True
    assert reached_assert


@pytest.mark.asyncio
async def test_data_stream_returns_data_from_multiple_event_messages():
    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, TEST_BUFFER_SIZE, SHORT_TEST_INTERVAL)
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

    reached_asserts = False
    async for data in _data_stream(buffer,
                                   queue,
                                   "broker", ["topic"],
                                   SHORT_TEST_INTERVAL,
                                   query_consumer=FakeQueryConsumer(),
                                   consumer_type=FakeConsumer,
                                   max_iterations=1):
        expected_tofs = np.concatenate((first_tof, second_tof))
        assert np.allclose(data.coords['tof'].values, expected_tofs)
        expected_ids = np.concatenate(
            (first_detector_ids, second_detector_ids))
        assert np.array_equal(data.coords['detector_id'].values, expected_ids)
        reached_asserts = True
    assert reached_asserts


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
    run_info_topic = "fake_topic"
    reached_assert = False
    test_instrument_name = "DATA_STREAM_TEST"
    async for data in _data_stream(
            buffer,
            queue,
            "broker", [""],
            SHORT_TEST_INTERVAL,
            run_info_topic=run_info_topic,
            query_consumer=FakeQueryConsumer(test_instrument_name),
            consumer_type=FakeConsumer,
            max_iterations=0):
        assert data["instrument_name"].value == test_instrument_name
        reached_assert = True
    assert reached_assert


@pytest.mark.asyncio
async def test_error_raised_if_no_run_start_message_available():
    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, TEST_BUFFER_SIZE, SHORT_TEST_INTERVAL)
    run_info_topic = "fake_topic"
    test_instrument_name = "DATA_STREAM_TEST"
    # Low and high offset are the same value, indicates there are
    # no messages available in the partition
    low_and_high_offset = (0, 0)
    with pytest.raises(RunStartError):
        async for _ in _data_stream(buffer,
                                    queue,
                                    "broker", [""],
                                    SHORT_TEST_INTERVAL,
                                    run_info_topic=run_info_topic,
                                    query_consumer=FakeQueryConsumer(
                                        test_instrument_name,
                                        low_and_high_offset),
                                    consumer_type=FakeConsumer,
                                    max_iterations=0):
            pass


@pytest.mark.asyncio
async def test_error_if_both_topics_and_run_start_topic_not_specified():
    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, TEST_BUFFER_SIZE, SHORT_TEST_INTERVAL)
    # At least one of "topics" and "run_start_topic" must be specified
    with pytest.raises(ValueError):
        async for _ in _data_stream(buffer,
                                    queue,
                                    "broker",
                                    topics=None,
                                    interval=SHORT_TEST_INTERVAL,
                                    run_info_topic=None,
                                    query_consumer=FakeQueryConsumer(),
                                    consumer_type=FakeConsumer,
                                    max_iterations=0):
            pass


@pytest.mark.asyncio
async def test_specified_topics_override_run_start_message_topics():
    # If "topics" argument is specified then they should be used, even if
    # a run start topic is provided
    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, TEST_BUFFER_SIZE, SHORT_TEST_INTERVAL)
    test_topics = ["whiting", "snail", "porpoise"]
    topic_in_run_start_message = "test_topic"
    test_streams = [Stream("/entry/stream_1", topic_in_run_start_message)]
    query_consumer = FakeQueryConsumer(streams=test_streams)
    # At least one of "topics" and "run_start_topic" must be specified
    async for _ in _data_stream(buffer,
                                queue,
                                "broker",
                                topics=test_topics,
                                interval=SHORT_TEST_INTERVAL,
                                run_info_topic=None,
                                query_consumer=query_consumer,
                                consumer_type=FakeConsumer,
                                max_iterations=0):
        pass
    for topic in test_topics:
        assert topic in query_consumer.queried_topics
    assert topic_in_run_start_message not in query_consumer.queried_topics


@pytest.mark.asyncio
async def test_topics_from_run_start_message_used_if_topics_arg_not_specified(
):
    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, TEST_BUFFER_SIZE, SHORT_TEST_INTERVAL)
    topic_in_run_start_message = "test_topic"
    test_streams = [Stream("/entry/stream_1", topic_in_run_start_message)]
    query_consumer = FakeQueryConsumer(streams=test_streams)
    # At least one of "topics" and "run_start_topic" must be specified
    async for _ in _data_stream(buffer,
                                queue,
                                "broker",
                                topics=None,
                                interval=SHORT_TEST_INTERVAL,
                                run_info_topic="run_topic",
                                query_consumer=query_consumer,
                                consumer_type=FakeConsumer,
                                max_iterations=0):
        pass
    assert topic_in_run_start_message in query_consumer.queried_topics


@pytest.mark.asyncio
async def test_start_time_from_run_start_msg_not_used_if_start_now_specified():
    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, TEST_BUFFER_SIZE, SHORT_TEST_INTERVAL)
    topic_in_run_start_message = "test_topic"
    test_streams = [Stream("/entry/stream_1", topic_in_run_start_message)]
    test_start_time = "2021-04-28T15:09:49Z"
    query_consumer = FakeQueryConsumer(streams=test_streams,
                                       start_time=test_start_time)
    # At least one of "topics" and "run_start_topic" must be specified
    async for _ in _data_stream(buffer,
                                queue,
                                "broker",
                                start_at=StartTime.now,
                                topics=None,
                                interval=SHORT_TEST_INTERVAL,
                                run_info_topic="run_topic",
                                query_consumer=query_consumer,
                                consumer_type=FakeConsumer,
                                max_iterations=0):
        pass

    assert query_consumer.queried_timestamp != int(
        dateutil.parser.parse(test_start_time).timestamp() * 1000.)


@pytest.mark.asyncio
async def test_start_time_from_run_start_msg_used_if_requested():
    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, TEST_BUFFER_SIZE, SHORT_TEST_INTERVAL)
    topic_in_run_start_message = "test_topic"
    test_streams = [Stream("/entry/stream_1", topic_in_run_start_message)]
    test_start_time = "2021-04-28T15:09:49Z"
    query_consumer = FakeQueryConsumer(streams=test_streams,
                                       start_time=test_start_time)
    # At least one of "topics" and "run_start_topic" must be specified
    async for _ in _data_stream(buffer,
                                queue,
                                "broker",
                                start_at=StartTime.start_of_run,
                                topics=None,
                                interval=SHORT_TEST_INTERVAL,
                                run_info_topic="run_topic",
                                query_consumer=query_consumer,
                                consumer_type=FakeConsumer,
                                max_iterations=0):
        pass

    assert query_consumer.queried_timestamp == int(
        dateutil.parser.parse(test_start_time).timestamp() * 1000.)
