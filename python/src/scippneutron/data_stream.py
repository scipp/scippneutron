import time
from typing import List, Generator, Callable, Optional
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
    topics: List[str],
    buffer_size: int = 1048576,
    interval: sc.Variable = 2. * sc.units.s,
    run_info_topic: Optional[str] = None
) -> Generator[sc.Variable, None, None]:
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
    data_stream will be from the last available run start message in the topic
    """
    try:
        from ._streaming_consumer import create_consumers, KafkaQueryConsumer
        from ._streaming_data_buffer import StreamedDataBuffer
    except ImportError:
        raise ImportError(_missing_dependency_message)

    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, buffer_size, interval)
    config = {
        "bootstrap.servers": kafka_broker,
        "group.id": "consumer_group_name",
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
    }
    time_now_ms = int(time.time() * 1000)
    consumers = create_consumers(time_now_ms,
                                 topics,
                                 config,
                                 buffer.new_data,
                                 stop_at_end_of_partition=False)

    # Use "async for" as "yield from" cannot be used in an async function, see
    # https://www.python.org/dev/peps/pep-0525/#asynchronous-yield-from
    async for v in _data_stream(buffer, queue, consumers,
                                interval, run_info_topic,
                                KafkaQueryConsumer(kafka_broker)):
        yield v


async def _data_stream(
    buffer: "StreamedDataBuffer",  # noqa: F821
    queue: asyncio.Queue,
    consumers: List["KafkaConsumer"],  # noqa: F821
    interval: sc.Variable,
    run_info_topic: Optional[str] = None,
    query_consumer: Optional["KafkaQueryConsumer"] = None  # noqa: F821
) -> Generator[sc.Variable, None, None]:
    """
    Main implementation of data stream is extracted to this function so that
    fake consumers can be injected for unit tests
    """
    try:
        from ._streaming_consumer import (start_consumers,
                                          get_run_start_message)
    except ImportError:
        raise ImportError(_missing_dependency_message)

    if run_info_topic is not None:
        run_start_info = get_run_start_message(run_info_topic, query_consumer)
        yield _load_nexus_json(run_start_info.nexus_structure)

    start_consumers(consumers)
    buffer.start()
    interval_s = sc.to_unit(interval, 's').value

    # If we wait twice the expected interval and have not got
    # any new data in the queue then check if it is because all
    # the consumers have stopped, if so, we are done. Otherwise
    # it could just be that we have not received any new data.
    while not _consumers_all_stopped(consumers):
        try:
            new_data = await asyncio.wait_for(queue.get(),
                                              timeout=2 * interval_s)
            yield new_data
        except asyncio.TimeoutError:
            pass

    buffer.stop()


def start_stream(user_function: Callable) -> asyncio.Task:
    return asyncio.create_task(user_function())
