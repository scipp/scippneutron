from ._streaming_consumer import (create_consumers, start_consumers)
from ._streaming_data_buffer import StreamedDataBuffer
import time
from typing import List, Generator
from ._streaming_consumer import KafkaConsumer
import asyncio
import scipp as sc


def _consumers_all_stopped(consumers: List[KafkaConsumer]):
    for consumer in consumers:
        if not consumer.stopped:
            return False
    return True


async def stream_data(
        interval_s: float = 2.) -> Generator[sc.Variable, None, None]:
    """
    Periodically yields accumulated data from stream.
    If the buffer fills up more frequently than the set interval
    then data is yielded more frequently.
    :param interval_s: interval between yielding any new data
      collected from stream
    """
    queue = asyncio.Queue()
    buffer = StreamedDataBuffer(queue, interval_s=interval_s)
    config = {
        "bootstrap.servers": "localhost:9092",
        "group.id": "consumer_group_name",
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
    }
    time_now_ms = int(time.time() * 1000)
    topics = ["ISIS_Kafka_Event_events"]
    consumers = create_consumers(time_now_ms,
                                 topics,
                                 config,
                                 buffer.new_data,
                                 stop_at_end_of_partition=True)

    start_consumers(consumers)
    buffer.start()

    # If we wait twice the expected interval and haven't got
    # any new data in the queue then check if it is because all
    # the consumers have stopped, if so, we are done. Otherwise
    # it could just be that we have received any new data.
    while not _consumers_all_stopped(consumers):
        try:
            new_data = await asyncio.wait_for(queue.get(),
                                              timeout=2 * interval_s)
            yield new_data
        except asyncio.TimeoutError:
            pass

    buffer.stop()
