# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import datetime
import pytest
import scipp as sc
import platform
import warnings
from typing import List, Tuple, Optional, Union
import numpy as np
from .nexus_helpers import NexusBuilder, Stream, Log, EventData
import multiprocessing as mp
from scippneutron.data_streaming._consumer_type import ConsumerType
from scippneutron.data_streaming._warnings import (UnknownFlatbufferIdWarning,
                                                   BufferSizeWarning)
from cmath import isclose

if platform.system() == "Darwin":
    pytest.skip(
        "Streaming tests disabled on MacOS because they tend to hang"
        " indefinitely for unknown reasons",
        allow_module_level=True)

try:
    import streaming_data_types  # noqa: F401
    from confluent_kafka import TopicPartition, KafkaError  # noqa: F401
    from scippneutron.data_streaming.data_stream import _data_stream  # noqa: E402
    from streaming_data_types.eventdata_ev42 import serialise_ev42  # noqa: E402
    from streaming_data_types.run_start_pl72 import serialise_pl72
    from streaming_data_types.logdata_f142 import serialise_f142
    from streaming_data_types.timestamps_tdct import serialise_tdct
    from streaming_data_types.sample_environment_senv import serialise_senv
    from streaming_data_types.sample_environment_senv import Location
    from scippneutron.data_streaming._consumer import RunStartError
    from scippneutron.data_streaming.data_stream import StopTime
except ImportError:
    pytest.skip("Kafka or Serialisation module is unavailable", allow_module_level=True)


class FakeKafkaError:

    def __init__(self, kafka_error_code: int):
        self._error_code = kafka_error_code

    def code(self):
        return self._error_code

    def __str__(self):
        return f"FakeKafkaError: code {self._error_code}"


class FakeMessage:

    def __init__(self,
                 payload: bytes,
                 error_code: Optional[int] = None,
                 timestamp: int = 0):
        self._payload = payload
        self._error_code = error_code
        self._timestamp = timestamp

    def value(self):
        return self._payload

    def error(self) -> Optional[KafkaError]:
        if self._error_code is not None:
            return FakeKafkaError(self._error_code)
        return None

    def timestamp(self) -> Tuple[None, int]:
        return None, self._timestamp


class FakeQueryConsumer:

    def __init__(self,
                 instrument_name: str = "",
                 low_and_high_offset: Tuple[int, int] = (2, 10),
                 streams: List[Stream] = None,
                 start_time: Optional[Union[int, datetime.datetime]] = None,
                 stop_time: Optional[Union[int, datetime.datetime]] = None,
                 nexus_structure: Optional[str] = None):
        self._instrument_name = instrument_name
        self._low_and_high_offset = low_and_high_offset
        self._streams = streams
        self.queried_topics = []
        self.queried_timestamp = None
        self._start_time = start_time
        self._stop_time = stop_time
        if self._start_time is None:
            self._start_time = datetime.datetime.now()
        if nexus_structure is None:
            builder = NexusBuilder()
            builder.add_instrument(self._instrument_name)
            if self._streams is not None:
                for stream in self._streams:
                    builder.add_stream(stream)
            self._nexus_structure = builder.json_string
        else:
            self._nexus_structure = nexus_structure

    @staticmethod
    def assign(partitions: List[TopicPartition]):
        pass

    def get_watermark_offsets(self, partition: TopicPartition) -> Tuple[int, int]:
        return self._low_and_high_offset

    def get_topic_partitions(self,
                             topic: str,
                             offset: int = -1) -> List[TopicPartition]:
        self.queried_topics.append(topic)
        return [TopicPartition(topic, partition=0, offset=offset)]

    def poll(self, timeout=2.) -> FakeMessage:
        return FakeMessage(
            serialise_pl72("",
                           "",
                           start_time=self._start_time,
                           stop_time=self._stop_time,
                           nexus_structure=self._nexus_structure))

    def seek(self, partition: TopicPartition):
        pass

    def offsets_for_times(self, partitions: List[TopicPartition]):
        self.queried_timestamp = partitions[0].offset
        return partitions


# Short time to use for buffer emit and data_stream interval in tests
# pass or fail fast!
SHORT_TEST_INTERVAL = 100. * sc.Unit('milliseconds')
# Small buffer of 20 events is sufficient for the tests
TEST_BUFFER_SIZE = 20

TEST_STREAM_ARGS = {
    "kafka_broker": "broker",
    "topics": ["topic"],
    "interval": SHORT_TEST_INTERVAL,
    "event_buffer_size": TEST_BUFFER_SIZE,
    "slow_metadata_buffer_size": TEST_BUFFER_SIZE,
    "fast_metadata_buffer_size": TEST_BUFFER_SIZE,
    "chopper_buffer_size": TEST_BUFFER_SIZE,
    "consumer_type": ConsumerType.FAKE,
    "timeout": 10. * sc.units.s
}
# "timeout" arg: if something gets broken then this makes sure the
# test will not get stuck in the _data_stream loop indefinitely.
# A TimeoutError is raised by _data_stream if the timeout occurs.


@pytest.fixture(scope="function")
def queues():
    # Specify to start the process using the "spawn" method, otherwise
    # on Linux the default is to fork the Python interpreter which
    # is "problematic" in a multithreaded process, in our case the use of
    # asyncio means the process is multithreaded.
    # See documentation:
    # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    ctx = mp.get_context("spawn")
    return ctx.Queue(), ctx.Queue(), ctx.Queue()


@pytest.mark.asyncio
async def test_data_stream_returns_data_from_single_event_message(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues
    time_of_flight = np.array([1., 2., 3.])
    detector_ids = np.array([4, 5, 6])
    test_message = FakeMessage(
        serialise_ev42("detector", 0, 0, time_of_flight, detector_ids))
    test_message_queue.put(test_message)

    reached_assert = False
    async for data in _data_stream(data_queue,
                                   worker_instruction_queue,
                                   halt_after_n_data_chunks=1,
                                   test_message_queue=test_message_queue,
                                   query_consumer=FakeQueryConsumer(),
                                   **TEST_STREAM_ARGS):
        assert np.allclose(data.coords['tof'].values, time_of_flight)
        reached_assert = True
    assert reached_assert


@pytest.mark.asyncio
async def test_data_stream_returns_data_from_multiple_event_messages(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues
    first_tof = np.array([1., 2., 3.])
    first_detector_ids = np.array([4, 5, 6])
    first_test_message = serialise_ev42("detector", 0, 0, first_tof, first_detector_ids)
    second_tof = np.array([1., 2., 3.])
    second_detector_ids = np.array([4, 5, 6])
    second_test_message = serialise_ev42("detector", 0, 0, second_tof,
                                         second_detector_ids)
    test_message_queue.put(FakeMessage(first_test_message))
    test_message_queue.put(FakeMessage(second_test_message))

    reached_asserts = False
    async for data in _data_stream(data_queue,
                                   worker_instruction_queue,
                                   halt_after_n_data_chunks=1,
                                   test_message_queue=test_message_queue,
                                   query_consumer=FakeQueryConsumer(),
                                   **TEST_STREAM_ARGS):
        expected_tofs = np.concatenate((first_tof, second_tof))
        assert np.allclose(data.coords['tof'].values, expected_tofs)
        expected_ids = np.concatenate((first_detector_ids, second_detector_ids))
        assert np.array_equal(data.coords['detector_id'].values, expected_ids)
        reached_asserts = True
    assert reached_asserts


@pytest.mark.asyncio
async def test_warn_if_unrecognised_message_was_encountered(queues):
    warnings.filterwarnings("error")
    data_queue, worker_instruction_queue, test_message_queue = queues
    # First 4 bytes of the message payload are the FlatBuffer schema identifier
    # "abcd" does not correspond to a FlatBuffer schema for data
    # that scipp is interested in
    test_message = b"abcd0000"

    with pytest.warns(UnknownFlatbufferIdWarning):
        test_message_queue.put(FakeMessage(test_message))
        async for _ in _data_stream(data_queue,
                                    worker_instruction_queue,
                                    halt_after_n_warnings=1,
                                    test_message_queue=test_message_queue,
                                    query_consumer=FakeQueryConsumer(),
                                    **TEST_STREAM_ARGS):
            test_message_queue.put(FakeMessage(test_message))


@pytest.mark.asyncio
async def test_warn_on_buffer_size_exceeded_by_single_message(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues
    buffer_size_2_events = 2
    time_of_flight = np.array([1., 2., 3.])
    detector_ids = np.array([4, 5, 6])
    test_message = serialise_ev42("detector", 0, 0, time_of_flight, detector_ids)

    test_steam_args = TEST_STREAM_ARGS.copy()
    test_steam_args["event_buffer_size"] = buffer_size_2_events

    with pytest.warns(BufferSizeWarning):
        test_message_queue.put(FakeMessage(test_message))
        async for _ in _data_stream(data_queue,
                                    worker_instruction_queue,
                                    halt_after_n_warnings=1,
                                    test_message_queue=test_message_queue,
                                    query_consumer=FakeQueryConsumer(),
                                    **test_steam_args):
            test_message_queue.put(FakeMessage(test_message))


@pytest.mark.asyncio
async def test_data_returned_when_buffer_size_exceeded_by_event_messages(queues):
    # Messages cumulatively exceed the buffer size, data_stream
    # will return multiple chunks of data to clear the buffer
    # between messages.
    data_queue, worker_instruction_queue, test_message_queue = queues
    first_tof = np.array([1., 2., 3.])
    first_detector_ids = np.array([4, 5, 6])
    first_test_message = serialise_ev42("detector", 0, 0, first_tof, first_detector_ids)
    second_tof = np.array([7., 8., 9.])
    second_detector_ids = np.array([4, 5, 6])
    second_test_message = serialise_ev42("detector", 0, 0, second_tof,
                                         second_detector_ids)

    # Event data buffer size is 5, so the second message
    # will not fit in the buffer
    test_stream_args = TEST_STREAM_ARGS.copy()
    test_stream_args["event_buffer_size"] = 5
    test_stream_args["run_info_topic"] = "run_info_topic"

    reached_asserts = False
    n_chunks = 0
    async for data in _data_stream(data_queue,
                                   worker_instruction_queue,
                                   halt_after_n_data_chunks=3,
                                   test_message_queue=test_message_queue,
                                   query_consumer=FakeQueryConsumer(),
                                   **test_stream_args):
        # n_chunks == 0 zeroth chunk contains data
        # from run start message
        if n_chunks == 0:
            test_message_queue.put(FakeMessage(first_test_message))
            test_message_queue.put(FakeMessage(second_test_message))
        elif n_chunks == 1:
            # Contain event data from first message
            assert np.allclose(data.coords['tof'].values, first_tof)
        elif n_chunks == 2:
            # Contain event data from second message
            assert np.allclose(data.coords['tof'].values, second_tof)
            reached_asserts = True

        n_chunks += 1
    assert reached_asserts


@pytest.mark.asyncio
async def test_data_are_loaded_from_run_start_message(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues
    run_info_topic = "fake_topic"
    reached_assert = False
    test_instrument_name = "DATA_STREAM_TEST"
    async for data in _data_stream(
            data_queue,
            worker_instruction_queue,
            run_info_topic=run_info_topic,
            halt_after_n_data_chunks=0,
            test_message_queue=test_message_queue,
            query_consumer=FakeQueryConsumer(test_instrument_name),
            **TEST_STREAM_ARGS):
        assert data["instrument_name"].value == test_instrument_name
        reached_assert = True
    assert reached_assert


@pytest.mark.asyncio
async def test_error_raised_if_no_run_start_message_available(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues
    run_info_topic = "fake_topic"
    test_instrument_name = "DATA_STREAM_TEST"

    # Low and high offset are the same value, indicates there are
    # no messages available in the partition
    low_and_high_offset = (0, 0)
    with pytest.raises(RunStartError):
        async for _ in _data_stream(data_queue,
                                    worker_instruction_queue,
                                    run_info_topic=run_info_topic,
                                    halt_after_n_data_chunks=0,
                                    test_message_queue=test_message_queue,
                                    query_consumer=FakeQueryConsumer(
                                        test_instrument_name, low_and_high_offset),
                                    **TEST_STREAM_ARGS):
            pass


@pytest.mark.asyncio
async def test_error_if_both_topics_and_run_start_topic_not_specified(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues

    test_stream_args = TEST_STREAM_ARGS.copy()
    test_stream_args["topics"] = None
    # At least one of "topics" and "run_start_topic" must be specified
    with pytest.raises(ValueError):
        async for _ in _data_stream(data_queue,
                                    worker_instruction_queue,
                                    run_info_topic=None,
                                    halt_after_n_data_chunks=0,
                                    **test_stream_args,
                                    query_consumer=FakeQueryConsumer(),
                                    test_message_queue=test_message_queue):
            pass


@pytest.mark.asyncio
async def test_specified_topics_override_run_start_message_topics(queues):
    # If "topics" argument is specified then they should be used, even if
    # a run start topic is provided
    data_queue, worker_instruction_queue, test_message_queue = queues
    test_topics = ["whiting", "snail", "porpoise"]
    topic_in_run_start_message = "test_topic"
    test_streams = [Stream("/entry", topic_in_run_start_message)]
    query_consumer = FakeQueryConsumer(streams=test_streams)
    test_stream_args = TEST_STREAM_ARGS.copy()
    test_stream_args["topics"] = test_topics
    async for _ in _data_stream(data_queue,
                                worker_instruction_queue,
                                run_info_topic=None,
                                query_consumer=query_consumer,
                                halt_after_n_data_chunks=0,
                                **test_stream_args,
                                test_message_queue=test_message_queue):
        pass
    assert not query_consumer.queried_topics, "Expected specified topics" \
                                              " to be used and none queried"


@pytest.mark.asyncio
async def test_data_stream_returns_metadata(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues
    run_info_topic = "fake_topic"
    test_instrument_name = "DATA_STREAM_TEST"

    # The Kafka topics to get metadata from are recorded as "stream" objects in
    # the nexus_structure field of the run start message
    # There are currently 3 schemas for metadata, they have flatbuffer ids
    # f142, senv and tdct
    f142_source_name = "f142_source"
    f142_log_name = "f142_log"
    senv_source_name = "senv_source"
    senv_log_name = "senv_log"
    tdct_source_name = "tdct_source"
    tdct_log_name = "tdct_log"
    streams = [
        Stream(f"/entry/{f142_log_name}", "f142_topic", f142_source_name, "f142",
               "double", "m"),
        Stream(f"/entry/{senv_log_name}", "senv_topic", senv_source_name, "senv",
               "double", "m"),
        Stream(f"/entry/{tdct_log_name}", "tdct_topic", tdct_source_name, "tdct")
    ]

    test_stream_args = TEST_STREAM_ARGS.copy()
    test_stream_args["topics"] = None
    n_chunks = 0
    async for data in _data_stream(data_queue,
                                   worker_instruction_queue,
                                   run_info_topic=run_info_topic,
                                   query_consumer=FakeQueryConsumer(
                                       test_instrument_name, streams=streams),
                                   halt_after_n_data_chunks=2,
                                   **test_stream_args,
                                   test_message_queue=test_message_queue):
        data_from_stream = data

        if n_chunks == 0:
            # Fake receiving a Kafka message for each metadata schema
            # Do this after the run start message has been parsed, so that
            # a metadata buffer will have been created for each data source
            # described in the start message.
            f142_value = 26.1236
            f142_timestamp = 123456  # ns after epoch
            f142_test_message = serialise_f142(f142_value, f142_source_name,
                                               f142_timestamp)
            test_message_queue.put(FakeMessage(f142_test_message))
            senv_values = np.array([26, 127, 52])
            senv_timestamp_ns = 123000  # ns after epoch
            senv_timestamp = datetime.datetime.fromtimestamp(senv_timestamp_ns * 1e-9,
                                                             datetime.timezone.utc)
            senv_time_between_samples = 100  # ns
            senv_test_message = serialise_senv(senv_source_name, -1, senv_timestamp,
                                               senv_time_between_samples, 0,
                                               senv_values, Location.Start)
            test_message_queue.put(FakeMessage(senv_test_message))
            tdct_timestamps = np.array([1234, 2345, 3456])  # ns
            tdct_test_message = serialise_tdct(tdct_source_name, tdct_timestamps)
            test_message_queue.put(FakeMessage(tdct_test_message))

        n_chunks += 1

    assert isclose(data_from_stream.attrs[f142_source_name].value.values[0], f142_value)
    assert data_from_stream.attrs[f142_source_name].value.coords['time'].values[
        0] == np.array(f142_timestamp, dtype=np.dtype('datetime64[ns]'))
    assert np.array_equal(data_from_stream.attrs[senv_source_name].value.values,
                          senv_values)
    senv_expected_timestamps = np.array([
        senv_timestamp_ns, senv_timestamp_ns + senv_time_between_samples,
        senv_timestamp_ns + (2 * senv_time_between_samples)
    ],
                                        dtype=np.dtype('datetime64[ns]'))
    assert np.array_equal(
        data_from_stream.attrs[senv_source_name].value.coords['time'].values,
        senv_expected_timestamps)
    assert np.array_equal(data_from_stream.attrs[tdct_source_name].value.values,
                          tdct_timestamps)


@pytest.mark.asyncio
async def test_data_stream_returns_data_from_multiple_slow_metadata_messages(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues
    run_info_topic = "fake_topic"
    test_instrument_name = "DATA_STREAM_TEST"

    # The Kafka topics to get metadata from are recorded as "stream" objects in
    # the nexus_structure field of the run start message
    f142_source_name = "f142_source"
    f142_log_name = "f142_log"
    streams = [
        Stream(f"/entry/{f142_log_name}", "f142_topic", f142_source_name, "f142",
               "double", "m"),
    ]

    test_stream_args = TEST_STREAM_ARGS.copy()
    test_stream_args["topics"] = None
    n_chunks = 0
    async for data in _data_stream(data_queue,
                                   worker_instruction_queue,
                                   run_info_topic=run_info_topic,
                                   query_consumer=FakeQueryConsumer(
                                       test_instrument_name, streams=streams),
                                   halt_after_n_data_chunks=2,
                                   **test_stream_args,
                                   test_message_queue=test_message_queue):
        data_from_stream = data

        if n_chunks == 0:
            # Fake receiving a Kafka message for each metadata schema
            # Do this after the run start message has been parsed, so that
            # a metadata buffer will have been created for each data source
            # described in the start message.
            f142_value_1 = 26.1236
            f142_timestamp_1 = 123456  # ns after epoch
            f142_test_message = serialise_f142(f142_value_1, f142_source_name,
                                               f142_timestamp_1)
            test_message_queue.put(FakeMessage(f142_test_message))
            f142_value_2 = 2.725
            f142_timestamp_2 = 234567  # ns after epoch
            f142_test_message = serialise_f142(f142_value_2, f142_source_name,
                                               f142_timestamp_2)
            test_message_queue.put(FakeMessage(f142_test_message))

        n_chunks += 1

    assert np.allclose(data_from_stream.attrs[f142_source_name].value.values,
                       np.array([f142_value_1, f142_value_2]))
    assert np.array_equal(
        data_from_stream.attrs[f142_source_name].value.coords['time'].values,
        np.array([f142_timestamp_1, f142_timestamp_2],
                 dtype=np.dtype('datetime64[ns]')))


@pytest.mark.asyncio
async def test_data_stream_returns_data_from_multiple_fast_metadata_messages(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues
    run_info_topic = "fake_topic"
    test_instrument_name = "DATA_STREAM_TEST"

    # The Kafka topics to get metadata from are recorded as "stream" objects in
    # the nexus_structure field of the run start message
    senv_source_name = "senv_source"
    senv_log_name = "senv_log"
    streams = [
        Stream(f"/entry/{senv_log_name}", "senv_topic", senv_source_name, "senv",
               "double", "m"),
    ]

    test_stream_args = TEST_STREAM_ARGS.copy()
    test_stream_args["topics"] = None
    n_chunks = 0
    async for data in _data_stream(data_queue,
                                   worker_instruction_queue,
                                   run_info_topic=run_info_topic,
                                   query_consumer=FakeQueryConsumer(
                                       test_instrument_name, streams=streams),
                                   halt_after_n_data_chunks=2,
                                   **test_stream_args,
                                   test_message_queue=test_message_queue):
        data_from_stream = data

        if n_chunks == 0:
            # Fake receiving a Kafka message for each metadata schema
            # Do this after the run start message has been parsed, so that
            # a metadata buffer will have been created for each data source
            # described in the start message.
            senv_values_1 = np.array([26, 127, 52])
            senv_timestamp_ns_1 = 123000  # ns after epoch
            senv_timestamp = datetime.datetime.fromtimestamp(senv_timestamp_ns_1 * 1e-9,
                                                             datetime.timezone.utc)
            senv_time_between_samples = 100  # ns
            senv_test_message = serialise_senv(senv_source_name, -1, senv_timestamp,
                                               senv_time_between_samples, 0,
                                               senv_values_1, Location.Start)
            test_message_queue.put(FakeMessage(senv_test_message))
            senv_values_2 = np.array([3832, 324, 3])
            senv_timestamp_ns_2 = 234000  # ns after epoch
            senv_timestamp = datetime.datetime.fromtimestamp(senv_timestamp_ns_2 * 1e-9,
                                                             datetime.timezone.utc)
            senv_test_message = serialise_senv(senv_source_name, -1, senv_timestamp,
                                               senv_time_between_samples, 0,
                                               senv_values_2, Location.Start)
            test_message_queue.put(FakeMessage(senv_test_message))

        n_chunks += 1

    assert np.array_equal(data_from_stream.attrs[senv_source_name].value.values,
                          np.concatenate((senv_values_1, senv_values_2)))
    senv_expected_timestamps_1 = np.array([
        senv_timestamp_ns_1, senv_timestamp_ns_1 + senv_time_between_samples,
        senv_timestamp_ns_1 + (2 * senv_time_between_samples)
    ],
                                          dtype=np.dtype('datetime64[ns]'))
    senv_expected_timestamps_2 = np.array([
        senv_timestamp_ns_2, senv_timestamp_ns_2 + senv_time_between_samples,
        senv_timestamp_ns_2 + (2 * senv_time_between_samples)
    ],
                                          dtype=np.dtype('datetime64[ns]'))
    assert np.array_equal(
        data_from_stream.attrs[senv_source_name].value.coords['time'].values,
        np.concatenate((senv_expected_timestamps_1, senv_expected_timestamps_2)))


@pytest.mark.asyncio
async def test_data_stream_returns_data_from_multiple_chopper_messages(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues
    run_info_topic = "fake_topic"
    test_instrument_name = "DATA_STREAM_TEST"

    # The Kafka topics to get metadata from are recorded as "stream" objects in
    # the nexus_structure field of the run start message
    tdct_source_name = "tdct_source"
    tdct_log_name = "tdct_log"
    streams = [
        Stream(f"/entry/{tdct_log_name}", "tdct_topic", tdct_source_name, "tdct")
    ]

    test_stream_args = TEST_STREAM_ARGS.copy()
    test_stream_args["topics"] = None
    n_chunks = 0
    async for data in _data_stream(data_queue,
                                   worker_instruction_queue,
                                   run_info_topic=run_info_topic,
                                   query_consumer=FakeQueryConsumer(
                                       test_instrument_name, streams=streams),
                                   halt_after_n_data_chunks=2,
                                   **test_stream_args,
                                   test_message_queue=test_message_queue):
        data_from_stream = data

        if n_chunks == 0:
            # Fake receiving a Kafka message for each metadata schema
            # Do this after the run start message has been parsed, so that
            # a metadata buffer will have been created for each data source
            # described in the start message.
            tdct_timestamps_1 = np.array([1234, 2345, 3456])  # ns
            tdct_test_message = serialise_tdct(tdct_source_name, tdct_timestamps_1)
            test_message_queue.put(FakeMessage(tdct_test_message))
            tdct_timestamps_2 = np.array([4567, 5678, 6789])  # ns
            tdct_test_message = serialise_tdct(tdct_source_name, tdct_timestamps_2)
            test_message_queue.put(FakeMessage(tdct_test_message))

        n_chunks += 1

    assert np.array_equal(data_from_stream.attrs[tdct_source_name].value.values,
                          np.concatenate((tdct_timestamps_1, tdct_timestamps_2)))


@pytest.mark.asyncio
async def test_data_stream_warns_if_fast_metadata_message_exceeds_buffer(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues
    buffer_size = 2
    run_info_topic = "fake_topic"
    test_instrument_name = "DATA_STREAM_TEST"

    # The Kafka topics to get metadata from are recorded as "stream" objects in
    # the nexus_structure field of the run start message
    senv_source_name = "senv_source"
    senv_log_name = "senv_log"
    streams = [
        Stream(f"/entry/{senv_log_name}", "senv_topic", senv_source_name, "senv",
               "double", "m"),
    ]

    test_stream_args = TEST_STREAM_ARGS.copy()
    test_stream_args["topics"] = None
    test_stream_args["fast_metadata_buffer_size"] = buffer_size
    with pytest.warns(BufferSizeWarning):
        async for _ in _data_stream(data_queue,
                                    worker_instruction_queue,
                                    run_info_topic=run_info_topic,
                                    query_consumer=FakeQueryConsumer(
                                        test_instrument_name, streams=streams),
                                    halt_after_n_warnings=1,
                                    **test_stream_args,
                                    test_message_queue=test_message_queue):
            # Fake receiving a Kafka message for each metadata schema
            # Do this after the run start message has been parsed, so that
            # a metadata buffer will have been created for each data source
            # described in the start message.

            # 3 values but buffer size is only 2!
            senv_values = np.array([26, 127, 52])
            senv_timestamp_ns = 123000  # ns after epoch
            senv_timestamp = datetime.datetime.fromtimestamp(senv_timestamp_ns * 1e-9,
                                                             datetime.timezone.utc)
            senv_time_between_samples = 100  # ns
            senv_test_message = serialise_senv(senv_source_name, -1, senv_timestamp,
                                               senv_time_between_samples, 0,
                                               senv_values, Location.Start)

            test_message_queue.put(FakeMessage(senv_test_message))


@pytest.mark.asyncio
async def test_data_stream_warns_if_single_chopper_message_exceeds_buffer(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues
    buffer_size = 2
    run_info_topic = "fake_topic"
    test_instrument_name = "DATA_STREAM_TEST"

    # The Kafka topics to get metadata from are recorded as "stream" objects in
    # the nexus_structure field of the run start message
    tdct_source_name = "tdct_source"
    tdct_log_name = "tdct_log"
    streams = [
        Stream(f"/entry/{tdct_log_name}", "tdct_topic", tdct_source_name, "tdct")
    ]

    test_stream_args = TEST_STREAM_ARGS.copy()
    test_stream_args["topics"] = None
    test_stream_args["chopper_buffer_size"] = buffer_size
    with pytest.warns(BufferSizeWarning):
        async for _ in _data_stream(data_queue,
                                    worker_instruction_queue,
                                    run_info_topic=run_info_topic,
                                    query_consumer=FakeQueryConsumer(
                                        test_instrument_name, streams=streams),
                                    halt_after_n_warnings=1,
                                    **test_stream_args,
                                    test_message_queue=test_message_queue):
            # Fake receiving a Kafka message for each metadata schema
            # Do this after the run start message has been parsed, so that
            # a metadata buffer will have been created for each data source
            # described in the start message.

            # 3 values but buffer size is only 2!
            tdct_timestamps = np.array([1234, 2345, 3456])  # ns
            tdct_test_message = serialise_tdct(tdct_source_name, tdct_timestamps)

            test_message_queue.put(FakeMessage(tdct_test_message))


@pytest.mark.asyncio
async def test_data_returned_if_multiple_slow_metadata_msgs_exceed_buffer(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues
    run_info_topic = "fake_topic"
    test_instrument_name = "DATA_STREAM_TEST"

    # The Kafka topics to get metadata from are recorded as "stream" objects in
    # the nexus_structure field of the run start message
    f142_source_name = "f142_source"
    f142_log_name = "f142_log"
    streams = [
        Stream(f"/entry/{f142_log_name}", "f142_topic", f142_source_name, "f142",
               "double", "m"),
    ]

    first_f142_value = 26.1236
    f142_timestamp = 123456  # ns after epoch
    first_message = serialise_f142(first_f142_value, f142_source_name, f142_timestamp)
    second_f142_value = 62.721
    second_message = serialise_f142(second_f142_value, f142_source_name, f142_timestamp)

    test_stream_args = TEST_STREAM_ARGS.copy()
    test_stream_args["slow_metadata_buffer_size"] = 1
    test_stream_args["topics"] = None
    n_chunks = 0
    reached_asserts = False
    async for data in _data_stream(data_queue,
                                   worker_instruction_queue,
                                   run_info_topic=run_info_topic,
                                   query_consumer=FakeQueryConsumer(
                                       test_instrument_name, streams=streams),
                                   halt_after_n_data_chunks=3,
                                   **test_stream_args,
                                   test_message_queue=test_message_queue):
        # n_chunks == 0 zeroth chunk contains data
        # from run start message
        if n_chunks == 0:
            test_message_queue.put(FakeMessage(first_message))
            test_message_queue.put(FakeMessage(second_message))
        elif n_chunks == 1:
            # Contains data from first message
            assert isclose(data.attrs[f142_source_name].value.values[0],
                           first_f142_value)
        elif n_chunks == 2:
            # Contains data from second message
            assert isclose(data.attrs[f142_source_name].value.values[0],
                           second_f142_value)
            reached_asserts = True
        n_chunks += 1

    assert reached_asserts


@pytest.mark.asyncio
async def test_data_returned_if_multiple_fast_metadata_msgs_exceed_buffer(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues
    buffer_size = 4
    run_info_topic = "fake_topic"
    test_instrument_name = "DATA_STREAM_TEST"

    # The Kafka topics to get metadata from are recorded as "stream" objects in
    # the nexus_structure field of the run start message
    senv_source_name = "senv_source"
    senv_log_name = "senv_log"
    streams = [
        Stream(f"/entry/{senv_log_name}", "senv_topic", senv_source_name, "senv",
               "double", "m"),
    ]

    first_senv_values = np.array([26, 127, 52])
    second_senv_values = np.array([72, 94, 1])
    senv_timestamp_ns = 123000  # ns after epoch
    senv_timestamp = datetime.datetime.fromtimestamp(senv_timestamp_ns * 1e-9,
                                                     datetime.timezone.utc)
    senv_time_between_samples = 100  # ns
    first_message = serialise_senv(senv_source_name, -1, senv_timestamp,
                                   senv_time_between_samples, 0, first_senv_values,
                                   Location.Start)
    second_message = serialise_senv(senv_source_name, -1, senv_timestamp,
                                    senv_time_between_samples, 0, second_senv_values,
                                    Location.Start)

    test_stream_args = TEST_STREAM_ARGS.copy()
    test_stream_args["topics"] = None
    test_stream_args["fast_metadata_buffer_size"] = buffer_size
    n_chunks = 0
    reached_asserts = False
    async for data in _data_stream(data_queue,
                                   worker_instruction_queue,
                                   run_info_topic=run_info_topic,
                                   query_consumer=FakeQueryConsumer(
                                       test_instrument_name, streams=streams),
                                   halt_after_n_data_chunks=3,
                                   **test_stream_args,
                                   test_message_queue=test_message_queue):
        # n_chunks == 0 zeroth chunk contains data
        # from run start message
        if n_chunks == 0:
            test_message_queue.put(FakeMessage(first_message))
            test_message_queue.put(FakeMessage(second_message))
        elif n_chunks == 1:
            # Contains data from first message
            assert np.array_equal(data.attrs[senv_source_name].value.values,
                                  first_senv_values)
        elif n_chunks == 2:
            # Contains data from second message
            assert np.array_equal(data.attrs[senv_source_name].value.values,
                                  second_senv_values)
            reached_asserts = True
        n_chunks += 1
    assert reached_asserts


@pytest.mark.asyncio
async def test_data_returned_if_multiple_chopper_msgs_exceed_buffer(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues
    buffer_size = 4
    run_info_topic = "fake_topic"
    test_instrument_name = "DATA_STREAM_TEST"

    # The Kafka topics to get metadata from are recorded as "stream" objects in
    # the nexus_structure field of the run start message
    tdct_source_name = "tdct_source"
    tdct_log_name = "tdct_log"
    streams = [
        Stream(f"/entry/{tdct_log_name}", "tdct_topic", tdct_source_name, "tdct")
    ]

    tdct_timestamps_1 = np.array([1234, 2345, 3456])  # ns
    first_tdct_message = serialise_tdct(tdct_source_name, tdct_timestamps_1)
    tdct_timestamps_2 = np.array([4567, 5678, 6789])  # ns
    second_tdct_message = serialise_tdct(tdct_source_name, tdct_timestamps_2)

    test_stream_args = TEST_STREAM_ARGS.copy()
    test_stream_args["topics"] = None
    test_stream_args["chopper_buffer_size"] = buffer_size

    n_chunks = 0
    reached_asserts = False
    async for data in _data_stream(data_queue,
                                   worker_instruction_queue,
                                   run_info_topic=run_info_topic,
                                   query_consumer=FakeQueryConsumer(
                                       test_instrument_name, streams=streams),
                                   **test_stream_args,
                                   halt_after_n_data_chunks=3,
                                   test_message_queue=test_message_queue):
        # n_chunks == 0 zeroth chunk contains data
        # from run start message
        if n_chunks == 0:
            test_message_queue.put(FakeMessage(first_tdct_message))
            test_message_queue.put(FakeMessage(second_tdct_message))
        elif n_chunks == 1:
            # Contains data from first message
            assert np.array_equal(data.attrs[tdct_source_name].value.values,
                                  tdct_timestamps_1)
        elif n_chunks == 2:
            # Contains data from second message
            assert np.array_equal(data.attrs[tdct_source_name].value.values,
                                  tdct_timestamps_2)
            reached_asserts = True
        n_chunks += 1
    assert reached_asserts


@pytest.mark.asyncio
async def test_no_warning_for_missing_datasets_if_group_contains_stream(queues):
    # Create NeXus description for run start message which contains
    # an NXlog which contains no datasets but does have a Stream
    # source for the data
    builder = NexusBuilder()
    test_instrument_name = "DATA_STREAM_TEST"
    builder.add_instrument(test_instrument_name)
    builder.add_log(Log("log", None))
    builder.add_event_data(EventData(None, None, None, None))
    builder.add_stream(Stream("/entry/log"))
    builder.add_stream(Stream("/entry/events_0"))
    nexus_structure = builder.json_string

    data_queue, worker_instruction_queue, test_message_queue = queues
    run_info_topic = "fake_topic"
    reached_assert = False

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        async for _ in _data_stream(data_queue,
                                    worker_instruction_queue,
                                    run_info_topic=run_info_topic,
                                    query_consumer=FakeQueryConsumer(
                                        test_instrument_name,
                                        nexus_structure=nexus_structure),
                                    **TEST_STREAM_ARGS,
                                    halt_after_n_data_chunks=0,
                                    test_message_queue=test_message_queue):
            reached_assert = True
            break
        assert reached_assert
        assert len(
            caught_warnings
        ) == 0, "Expect no 'missing datasets' warning from the NXlog or " \
                "NXevent_data because they each contain a stream which " \
                "will provide the missing data"


@pytest.mark.asyncio
async def test_data_stream_times_out(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues
    run_info_topic = "fake_topic"
    test_instrument_name = "DATA_STREAM_TEST"

    test_stream_args = TEST_STREAM_ARGS.copy()
    test_stream_args["topics"] = None
    # "timeout" is only for use in tests, ensures that if something
    # gets broken then tests will not get stuck in the _data_stream
    # loop indefinitely.
    test_stream_args["timeout"] = 2. * sc.units.s
    timed_out = False
    try:
        async for _ in _data_stream(
                data_queue,
                worker_instruction_queue,
                run_info_topic=run_info_topic,
                query_consumer=FakeQueryConsumer(test_instrument_name),
                **test_stream_args,
                test_message_queue=test_message_queue,
                end_at=StopTime.END_OF_RUN):
            # Do nothing until it times out
            pass
    except TimeoutError:
        timed_out = True
    assert timed_out


@pytest.mark.asyncio
async def test_stream_loop_exits_if_stop_time_and_end_of_partition_reached(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues
    run_info_topic = "fake_topic"
    test_instrument_name = "DATA_STREAM_TEST"

    test_stream_args = TEST_STREAM_ARGS.copy()
    test_stream_args["topics"] = None
    # System time is already after this stop time so the stream will stop
    # as soon as it sees the end of partition or a message with a
    # timestamp after the stop time
    stop_time_in_past = datetime.datetime(2017, 11, 28, 23, 55, 59, 342380)
    n_chunks = 0
    async for _ in _data_stream(data_queue,
                                worker_instruction_queue,
                                run_info_topic=run_info_topic,
                                query_consumer=FakeQueryConsumer(
                                    test_instrument_name, stop_time=stop_time_in_past),
                                **test_stream_args,
                                test_message_queue=test_message_queue,
                                end_at=StopTime.END_OF_RUN):
        if n_chunks == 0:
            # Tell consumer it has reached the end of the partition
            # Consumer will stop, data_consumption_manager will see the
            # consumer has stopped and stop the buffer and the
            # _data_stream will exit.
            # A TimeoutError would occur if the functionality is broken.
            test_message_queue.put(FakeMessage(b"", KafkaError._PARTITION_EOF))
        n_chunks += 1


@pytest.mark.asyncio
async def test_stream_loop_exits_if_stop_time_reached_and_later_message_seen(queues):
    data_queue, worker_instruction_queue, test_message_queue = queues
    run_info_topic = "fake_topic"
    test_instrument_name = "DATA_STREAM_TEST"

    # The Kafka topics to get metadata from are recorded as "stream" objects in
    # the nexus_structure field of the run start message
    f142_source_name = "f142_source"
    f142_log_name = "f142_log"
    streams = [
        Stream(f"/entry/{f142_log_name}", "f142_topic", f142_source_name, "f142",
               "double", "m"),
    ]

    test_stream_args = TEST_STREAM_ARGS.copy()
    test_stream_args["topics"] = None
    # System time is already after this stop time so the stream will stop
    # as soon as it sees the end of partition or a message with a
    # timestamp after the stop time
    stop_time_in_past = datetime.datetime(2017, 11, 28, 23, 55, 59, 342380)
    n_chunks = 0
    async for data in _data_stream(data_queue,
                                   worker_instruction_queue,
                                   run_info_topic=run_info_topic,
                                   query_consumer=FakeQueryConsumer(
                                       test_instrument_name,
                                       stop_time=stop_time_in_past,
                                       streams=streams),
                                   **test_stream_args,
                                   test_message_queue=test_message_queue,
                                   end_at=StopTime.END_OF_RUN):
        if n_chunks == 0:
            # Publish a message with a timestamp before the stop time
            f142_value_1 = 26.1236
            timestamp_before_stop_dt = datetime.datetime(2017, 11, 28, 23, 55, 50, 0)
            # Convert to integer nanoseconds
            # (for timestamp in message payload)
            timestamp_before_stop_ns = int(timestamp_before_stop_dt.timestamp() *
                                           1_000_000_000)
            # Convert to integer milliseconds
            # (for Kafka message header)
            timestamp_before_stop_ms = int(timestamp_before_stop_dt.timestamp() * 1_000)
            f142_test_message = serialise_f142(f142_value_1, f142_source_name,
                                               timestamp_before_stop_ns)
            test_message_queue.put(
                FakeMessage(f142_test_message, timestamp=timestamp_before_stop_ms))
        elif n_chunks == 1:
            # The data from the first message will be returned
            assert np.allclose(data.attrs[f142_source_name].value.values,
                               np.array([f142_value_1]))
            assert np.array_equal(
                data.attrs[f142_source_name].value.coords['time'].values,
                np.array([timestamp_before_stop_ns], dtype=np.dtype('datetime64[ns]')))

            # Publish message with timestamp after stop time, this will trigger
            # the consumer to stop and data_stream to exit.
            # A TimeoutError would occur if the functionality is broken.
            f142_value_2 = 2.725
            timestamp_after_stop_dt = datetime.datetime(2017, 11, 28, 23, 56, 50, 0)
            timestamp_after_stop_ns = int(timestamp_after_stop_dt.timestamp() *
                                          1_000_000_000)
            timestamp_after_stop_ms = int(timestamp_after_stop_dt.timestamp() * 1_000)
            f142_test_message = serialise_f142(f142_value_2, f142_source_name,
                                               timestamp_after_stop_ns)
            test_message_queue.put(
                FakeMessage(f142_test_message, timestamp=timestamp_after_stop_ms))

        n_chunks += 1
