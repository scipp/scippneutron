import numpy as np
import time
from typing import List, Generator, Optional, Set
import asyncio
import scipp as sc
from ..file_loading.load_nexus import _load_nexus_json, StreamInfo
import multiprocessing as mp
from queue import Empty as QueueEmpty
from enum import Enum
from ._consumer_type import ConsumerType
"""
Some type names are included as strings as imports are done in
function scope to avoid optional dependencies being imported
by the top level __init__.py
"""


class StartTime(Enum):
    now = "now"
    start_of_run = "start_of_run"


_missing_dependency_message = (
    "Confluent Kafka Python library and/or serialisation library"
    "not found, please install confluent-kafka and "
    "ess-streaming-data-types as detailed in the "
    "installation instructions (https://scipp.github.io/"
    "scippneutron/getting-started/installation.html)")


async def data_stream(
    kafka_broker: str,
    topics: Optional[List[str]] = None,
    event_buffer_size: int = 1_048_576,
    slow_metadata_buffer_size: int = 1000,
    fast_metadata_buffer_size: int = 100_000,
    chopper_buffer_size: int = 10_000,
    interval: sc.Variable = 2. * sc.units.s,
    run_info_topic: Optional[str] = None,
    start_time: StartTime = StartTime.now,
) -> Generator[sc.DataArray, None, None]:
    """
    Periodically yields accumulated data from stream.
    If the buffer fills up more frequently than the set interval
    then data is yielded more frequently.
    1048576 event buffer is around 24 MB (with pulse_time, id, weights, etc)
    :param kafka_broker: Address of the Kafka broker to stream data from
    :param topics: Kafka topics to consume data from
    :param event_buffer_size: Size of buffer to accumulate event data in
    :param slow_metadata_buffer_size: Size of buffer to accumulate slow
      sample env metadata in
    :param fast_metadata_buffer_size: Size of buffer to accumulate fast
      sample env metadata in
    :param chopper_buffer_size: Size of buffer to accumulate chopper
      timestamps in
    :param interval: interval between yielding any new data
      collected from stream
    :param run_info_topic: If provided, the first data batch returned by
      data_stream will be from the last available run start message in
      the topic
    :param start_time: Get data from now or from start of the last run
    """
    validate_buffer_size_args(chopper_buffer_size, event_buffer_size,
                              fast_metadata_buffer_size,
                              slow_metadata_buffer_size)

    data_queue = mp.Queue()
    instruction_queue = mp.Queue()

    # Use "async for" as "yield from" cannot be used in an async function, see
    # https://www.python.org/dev/peps/pep-0525/#asynchronous-yield-from
    async for data_chunk in _data_stream(
        data_queue, instruction_queue, kafka_broker, topics, interval,
        event_buffer_size, slow_metadata_buffer_size,
        fast_metadata_buffer_size, chopper_buffer_size, run_info_topic,
        start_time):  # noqa: E125
        yield data_chunk


def validate_buffer_size_args(chopper_buffer_size, event_buffer_size,
                              fast_metadata_buffer_size,
                              slow_metadata_buffer_size):
    for buffer_name, buffer_size in (("event_buffer_size", event_buffer_size),
                                     ("slow_metadata_buffer_size",
                                      slow_metadata_buffer_size),
                                     ("fast_metadata_buffer_size",
                                      fast_metadata_buffer_size),
                                     ("chopper_buffer_size",
                                      chopper_buffer_size)):
        if buffer_size < 1:
            raise ValueError(f"{buffer_name} must be greater than zero")


class WorkerInstruction(Enum):
    STOP_NOW = 1
    STOPTIME_UPDATE = 2


def data_consumption_worker(
        start_time_ms: int, topics: Set[str], kafka_broker: str,
        consumer_type: ConsumerType, stream_info: Optional[List[StreamInfo]],
        interval_s: float, event_buffer_size: int,
        slow_metadata_buffer_size: int, fast_metadata_buffer_size: int,
        chopper_buffer_size: int, worker_instruction_queue: mp.Queue,
        data_queue: mp.Queue):
    """
    Starts and stops buffers and data consumers which collect data and
    send them back to the main process via a queue.

    All input args must be mp.Queue or picklable as this function is launched
    as a multiprocessing.Process.
    """
    try:
        from ._consumer import (start_consumers, stop_consumers,
                                create_consumers)
        from ._data_buffer import StreamedDataBuffer
    except ImportError:
        raise ImportError(_missing_dependency_message)

    buffer = StreamedDataBuffer(data_queue, event_buffer_size,
                                slow_metadata_buffer_size,
                                fast_metadata_buffer_size, chopper_buffer_size,
                                interval_s)

    if stream_info is not None:
        buffer.init_metadata_buffers(stream_info)

    consumers = create_consumers(start_time_ms,
                                 topics,
                                 kafka_broker,
                                 consumer_type,
                                 buffer.new_data,
                                 stop_at_end_of_partition=False)

    start_consumers(consumers)
    buffer.start()

    while True:
        try:
            instruction = worker_instruction_queue.get(timeout=10.)
            if instruction == WorkerInstruction.STOP_NOW:
                stop_consumers(consumers)
                buffer.stop()
                break
        except QueueEmpty:
            pass
        except (ValueError, OSError):
            # Queue has been closed, stop worker
            stop_consumers(consumers)
            buffer.stop()
            break


async def _data_stream(
        data_queue: mp.Queue,
        worker_instruction_queue: mp.Queue,
        kafka_broker: str,
        topics: Optional[List[str]],
        interval: sc.Variable,
        event_buffer_size: int,
        slow_metadata_buffer_size: int,
        fast_metadata_buffer_size: int,
        chopper_buffer_size: int,
        run_info_topic: Optional[str] = None,
        start_at: StartTime = StartTime.now,
        query_consumer: Optional["KafkaQueryConsumer"] = None,  # noqa: F821
        consumer_type: ConsumerType = ConsumerType.REAL,
        max_iterations: int = np.iinfo(np.int32).max,  # for testability
) -> Generator[sc.DataArray, None, None]:
    """
    Main implementation of data stream is extracted to this function so that
    fake consumers can be injected for unit tests
    """
    try:
        from ._consumer import (get_run_start_message, KafkaQueryConsumer,
                                KafkaConsumer)
    except ImportError:
        raise ImportError(_missing_dependency_message)

    if topics is None and run_info_topic is None:
        raise ValueError("At least one of 'topics' and 'run_info_topic'"
                         " must be specified")

    # These are defaulted to None in the function signature
    # to avoid them having to be imported
    if query_consumer is None:
        query_consumer = KafkaQueryConsumer(kafka_broker)
    if consumer_type is None:
        consumer_type = KafkaConsumer

    stream_info = None
    if run_info_topic is not None:
        run_start_info = get_run_start_message(run_info_topic, query_consumer)
        if topics is None:
            loaded_data, stream_info = _load_nexus_json(
                run_start_info.nexus_structure, get_start_info=True)
            topics = {stream.topic for stream in stream_info}
        else:
            loaded_data, _ = _load_nexus_json(run_start_info.nexus_structure,
                                              get_start_info=False)
        yield loaded_data

    if start_at == StartTime.start_of_run:
        start_time = run_start_info.start_time * sc.Unit("milliseconds")
    else:
        start_time = time.time() * sc.units.s

    # Convert to int and float as easier to pass to mp.Process
    # (sc.Variable would have to be serialised/deserialised)
    start_time_ms = int(sc.to_unit(start_time, "milliseconds").value)
    interval_s = float(sc.to_unit(interval, 's').value)

    data_collect_process = mp.Process(
        target=data_consumption_worker,
        args=(start_time_ms, topics, kafka_broker, consumer_type, stream_info,
              interval_s, event_buffer_size, slow_metadata_buffer_size,
              fast_metadata_buffer_size, chopper_buffer_size,
              worker_instruction_queue, data_queue))
    data_collect_process.start()

    iterations = 0
    try:
        while data_collect_process.is_alive() and iterations < max_iterations:
            try:
                new_data = data_queue.get_nowait()
                iterations += 1
                yield new_data
            except QueueEmpty:
                await asyncio.sleep(0.5 * interval_s)
    finally:
        # Ensure cleanup happens however the loop exits
        worker_instruction_queue.put(WorkerInstruction.STOP_NOW)
        data_collect_process.join()
