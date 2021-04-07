from ._streaming_consumer import (create_consumers, start_consumers)
from ._streaming_data_buffer import StreamedDataBuffer
import time
from typing import List, Generator, Callable
from ._streaming_consumer import KafkaConsumer
import asyncio
import scipp as sc


def _consumers_all_stopped(consumers: List[KafkaConsumer]):
    for consumer in consumers:
        if not consumer.stopped:
            return False
    return True


async def _data_stream(
        buffer: StreamedDataBuffer,
        queue: asyncio.Queue,
        consumers: List[KafkaConsumer],
        interval_s: float = 2.) -> Generator[sc.Variable, None, None]:
    """
    Main implementation of data stream is extracted to this function so that
    fake consumers can be injected for unit tests
    """
    start_consumers(consumers)
    buffer.start()

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


async def data_stream(
        kafka_broker: str,
        topics: List[str],
        buffer_size: int = 1048576,
        interval_s: float = 2.) -> Generator[sc.Variable, None, None]:
    """
    Periodically yields accumulated data from stream.
    If the buffer fills up more frequently than the set interval
    then data is yielded more frequently.
    1048576 event buffer is around 24 MB (with pulse_time, id, weights, etc)
    :param kafka_broker: Address of the Kafka broker to stream data from
    :param topics: Kafka topics to consume data from
    :param buffer_size: Size of buffer to accumulate data in
    :param interval_s: interval between yielding any new data
      collected from stream
    """
    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, buffer_size, interval_s=interval_s)
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
    async for v in _data_stream(buffer, queue, consumers, interval_s):
        yield v


def start_stream(user_function: Callable) -> asyncio.Task:
    return asyncio.create_task(user_function())
