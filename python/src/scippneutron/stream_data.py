from ._streaming_consumer import (create_consumers, start_consumers,
                                  stop_consumers)
from ._streaming_data_buffer import StreamedDataBuffer
import time


def stream_data(interval_s=2.):
    """
    It periodically yields accumulated data and resets the buffer.
    If the buffer fills up within the emit time
    interval then data is emitted more frequently.
    :param interval_s:
    :return:
    """
    buffer = StreamedDataBuffer()
    config = {
        "bootstrap.servers": "localhost:9092",
        "group.id": "consumer_group_name",
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
    }
    # Real implementation would get the last "run start" message
    # to get the start timestamp and what other topics to consume from
    time_now_ms = int(time.time() * 1000)
    topics = ["ISIS_Kafka_Event_events"]
    consumers = create_consumers(time_now_ms,
                                 topics,
                                 config,
                                 buffer.new_data,
                                 stop_at_end_of_partition=True)

    start_consumers(consumers)
    buffer.start()

    time.sleep(10)

    stop_consumers(consumers)
    buffer.stop()
