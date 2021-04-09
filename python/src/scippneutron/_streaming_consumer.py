from confluent_kafka import Consumer, TopicPartition, KafkaError
from typing import Callable, List, Dict, Optional
import asyncio
from warnings import warn


class KafkaConsumer:
    def __init__(self, topic_partitions: List[TopicPartition], conf: Dict,
                 callback: Callable, stop_at_end_of_partition: bool):
        conf['enable.partition.eof'] = stop_at_end_of_partition
        self._consumer = Consumer(conf)
        self._consumer.assign(topic_partitions)
        self._callback = callback
        self._stop_at_end_of_partition = stop_at_end_of_partition
        self._reached_eop = False
        self._cancelled = False
        self._consume_data: Optional[asyncio.Task] = None
        self.stopped = True

    def start(self):
        self.stopped = False
        self._cancelled = False
        self._consume_data = asyncio.create_task(self._consume_loop())

    async def _consume_loop(self):
        while not self._cancelled:
            msg = self._consumer.poll(timeout=0.)
            if msg is None:
                await asyncio.sleep(0.2)
                continue
            if msg.error():
                if self._stop_at_end_of_partition and msg.error().code(
                ) == KafkaError._PARTITION_EOF:
                    self._reached_eop = True
                    break
                warn(f"Message error in consumer: {msg.error()}")
                break
            await self._callback(msg.value())
        self.stop()

    def stop(self):
        if not self._cancelled:
            self._cancelled = True
            if self._consume_data is not None:
                self._consume_data.cancel()
        self._consumer.close()
        self.stopped = True


def create_consumers(
        start_time_ms: int,
        topics: List[str],
        conf: Dict,
        callback: Callable,
        stop_at_end_of_partition: bool = False) -> List[KafkaConsumer]:
    """
    Creates one consumer per TopicPartition that start consuming
    at specified timestamp

    Having each consumer only be responsible for one partition
    greatly simplifies the logic around stopping at the end of
    the stream (making use of "end of partition" event)
    """
    consumer = Consumer(**conf)

    topic_partitions = []
    for topic in topics:
        metadata = consumer.list_topics(topic)
        topic_partitions.extend([
            TopicPartition(topic, partition[1].id, offset=start_time_ms)
            for partition in metadata.topics[topic].partitions.items()
        ])
    topic_partitions = consumer.offsets_for_times(topic_partitions)

    consumers = []
    for topic_partition in topic_partitions:
        consumers.append(
            KafkaConsumer([topic_partition], conf, callback,
                          stop_at_end_of_partition))

    return consumers


def start_consumers(consumers: List[KafkaConsumer]):
    for consumer in consumers:
        consumer.start()


def stop_consumers(consumers: List[KafkaConsumer]):
    for consumer in consumers:
        consumer.stop()
