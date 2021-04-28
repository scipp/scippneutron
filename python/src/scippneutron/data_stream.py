import numpy as np
import time
from typing import List, Generator, Callable, Optional, Type
import asyncio
import scipp as sc
from .load_nexus import _load_nexus_json
"""
Some type names are included as strings as imports are done in
function scope to avoid optional dependencies being imported
by the top level __init__.py
"""


def _consumers_all_stopped(consumers: List["KafkaConsumer"]):  # noqa: F821
    for consumer in consumers:
        if not consumer.stopped:
            return False
    return True


_missing_dependency_message = (
    "Confluent Kafka Python library and/or serialisation library"
    "not found, please install confluent-kafka and "
    "ess-streaming-data-types as detailed in the "
    "installation instructions (https://scipp.github.io/"
    "scippneutron/getting-started/installation.html)")


async def data_stream(
    kafka_broker: str,
    topics: Optional[List[str]] = None,
    buffer_size: int = 1048576,
    interval: sc.Variable = 2. * sc.units.s,
    run_info_topic: Optional[str] = None
) -> Generator[sc.DataArray, None, None]:
    """
    Periodically yields accumulated data from stream.
    If the buffer fills up more frequently than the set interval
    then data is yielded more frequently.
    1048576 event buffer is around 24 MB (with pulse_time, id, weights, etc)
    :param kafka_broker: Address of the Kafka broker to stream data from
    :param topics: Kafka topics to consume data from
    :param buffer_size: Size of buffer to accumulate data in
    :param interval: interval between yielding any new data
      collected from stream
    :param run_info_topic: If provided, the first data batch returned by
      data_stream will be from the last available run start message in
      the topic
    """
    try:
        from ._streaming_data_buffer import StreamedDataBuffer
    except ImportError:
        raise ImportError(_missing_dependency_message)

    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, buffer_size, interval)

    # Use "async for" as "yield from" cannot be used in an async function, see
    # https://www.python.org/dev/peps/pep-0525/#asynchronous-yield-from
    async for data_chunk in _data_stream(buffer, queue, kafka_broker, topics,
                                         interval, run_info_topic):
        yield data_chunk


async def _data_stream(
    buffer: "StreamedDataBuffer",  # noqa: F821
    queue: asyncio.Queue,
    kafka_broker: str,
    topics: Optional[List[str]],
    interval: sc.Variable,
    run_info_topic: Optional[str] = None,
    query_consumer: Optional["KafkaQueryConsumer"] = None,  # noqa: F821
    consumer_type: Optional[Type["KafkaConsumer"]] = None,  # noqa: F821
    max_iterations: int = np.iinfo(np.int32).max  # for testability
) -> Generator[sc.DataArray, None, None]:
    """
    Main implementation of data stream is extracted to this function so that
    fake consumers can be injected for unit tests
    """
    try:
        from ._streaming_consumer import (start_consumers, create_consumers,
                                          get_run_start_message,
                                          KafkaQueryConsumer, KafkaConsumer)
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

    start_time = time.time() * sc.units.s
    if run_info_topic is not None:
        run_start_info = get_run_start_message(run_info_topic, query_consumer)
        if topics is None:
            loaded_data, _, topics = _load_nexus_json(
                run_start_info.nexus_structure, get_start_info=True)
        else:
            loaded_data, _, _ = _load_nexus_json(
                run_start_info.nexus_structure, get_start_info=False)
        yield loaded_data

    consumers = create_consumers(start_time,
                                 topics,
                                 kafka_broker,
                                 query_consumer,
                                 consumer_type,
                                 buffer.new_data,
                                 stop_at_end_of_partition=False)

    start_consumers(consumers)
    buffer.start()

    # If we wait twice the expected interval and have not got
    # any new data in the queue then check if it is because all
    # the consumers have stopped, if so, we are done. Otherwise
    # it could just be that we have not received any new data.
    iterations = 0
    while not _consumers_all_stopped(
            consumers) and iterations < max_iterations:
        try:
            new_data = await asyncio.wait_for(queue.get(),
                                              timeout=2 *
                                              sc.to_unit(interval, 's').value)
            iterations += 1
            yield new_data
        except asyncio.TimeoutError:
            pass

    buffer.stop()


def start_stream(user_function: Callable) -> asyncio.Task:
    return asyncio.create_task(user_function())
